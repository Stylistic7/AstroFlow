"""Trace a cart item's pipeline and return every expected output path.

Under the pure-Merkle `content_hash := producer.call_hash` convention
(stamped into file headers by `@cached.remote`), filenames are deterministic
and statically computable from the trace alone — no file I/O at trace time
beyond ImageMetadata already in context.

The trace itself can be expensive (real measurement on a "PixInsight
Downscaled" mosaic with 730 nodes: ~17 s — dominated by
`resolve_calibration`'s filesystem-walking calibration matching), but it's
a deterministic PURE function of (pipeline source, entity, ProcessingConfig,
metadata), so it's never recomputed on the read path. `compute_graph.
_store_tap_snapshot` walks the graph once at build time and persists the
resolved taps into the `tap_snapshots` ContextActor dict, content-addressed
by `(pipeline_source_hash, stable_hash, content_hash, pipeline)`. Every
read — the web dashboard, tiles, the catalog — is then a ~0.1 ms
`context.get` of that snapshot (`_read_tap_snapshot`): no Ray, no trace.

A read that misses the snapshot means the object×pipeline was never built.
`recompute=False` callers (the viewer / catalog — see `tap_outputs_for`)
take the miss as "nothing on disk to show" and return `[]` rather than
tracing on an HTTP request. Only head-driver callers that legitimately need
a trace for something NOT in the current graph — `scripts.sweep_orphans`
(live set) and `state.archive` (cart→archive) — pass `recompute=True`, and
for them the miss recomputes the trace INLINE (pure, in-process). There is
deliberately no persistent trace cache beyond the build-time snapshot: the
trace is authoritative and cheap-to-recompute, so a second (diskcache) copy
would only add a consistency surface, head-pinning, and stale-read bugs to
keep in sync — see `feedback_avoid_derived_state`.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Any

from context import local_context as context


class TraceResolutionError(RuntimeError):
    """A cart item could NOT be traced to its output paths — the entity is
    absent from context, the pipeline is unregistered, or tracing raised.

    Distinct from a *successful* trace that legitimately yields no paths (an
    empty list). Callers that must establish a COMPLETE live set before a
    destructive step — the orphan sweep, cart→archive — pass `strict=True` and
    let this propagate rather than silently proceeding on a PARTIAL set. A
    swallowed trace failure looks identical to "this item has no outputs", so
    the GC would treat the item's real, on-disk files as orphans and delete
    them. Fail-fast: never delete (or archive, or run) against a trace we
    couldn't resolve.
    """


@dataclass(frozen=True)
class _TracedOutput:
    """In-memory proxy for an upstream Node's result at trace time.

    Carries `content_hash` (the producer's call_hash under the pure-Merkle
    convention) and `entity` (the dep's primary entity, used by polymorphic
    filename_fn dispatchers to route by identity).

    Deliberately NOT a `ContentHashable` subclass —
    `hashing._maybe_use_content_hash` then skips the `get_content_hash(file)`
    branch (which would read disk) and takes the `.content_hash` attribute
    branch instead. Bit-for-bit identical to what the runtime produces
    when it reads the stamped file header.

    `entity` is the dep's first identity-bearing arg (or recursive
    ancestor). When `downscale_stack(8, stack)` becomes a dep of
    `stack_files(stack_dep, config, *cal_masters)`, the dispatcher in
    `_make_dispatcher` sees ProcessingConfig at args[0] and needs
    `entity` on the `_TracedOutput` proxy to find the actual Stack.
    """
    content_hash: str
    entity: Any = None


def paths_for_cart_item(
    item: tuple[str, str, str], *, finals_only: bool = False,
    strict: bool = False,
) -> list[Path]:
    """Trace `item`'s pipeline against the current context and return
    every `filename_fn`-resolvable output path.

    `item` is the canonical cart-entry tuple `(object_type, object_id,
    pipeline_name)`. Returns `[]` if the entity isn't in context (stale
    cart entry) or the pipeline isn't registered.

    With `finals_only=True`, restricts to outputs of ops decorated
    `@node(... final=True)` — used by archive transition with
    `preserve="finals"`.

    With `strict=True` (the orphan sweep, cart→archive), a trace FAILURE —
    entity absent, pipeline unregistered, or tracing raised — raises
    `TraceResolutionError` instead of returning `[]`, so a caller building a
    live set before a destructive step never mistakes an unresolved trace for
    "no outputs". A successful-but-empty trace still returns `[]`.

    Reads the build-time snapshot (Ray-free, ~0.1 ms, via
    `traced_paths_for`); on a snapshot miss, `recompute=True` recomputes
    the trace inline.
    """
    return [Path(e["path"]) for e in traced_paths_for(item, strict=strict)
            if not finals_only or e["is_final"]]


def traced_paths_for(
    item: tuple[str, str, str],
    recompute: bool = True,
    ancestors: tuple[tuple[str, str], ...] = (),
    strict: bool = False,
) -> list[dict]:
    """Return the raw trace result for `item`: one dict per
    filename_fn-resolvable node, with keys `path`, `op_name`,
    `identity`, `is_final`.

    A ~0.1 ms snapshot read per call (no Ray on the read path). Use this
    when you need to look up multiple paths from the same cart item —
    hoist it out of any loop, iterate the returned list in pure Python.

    `paths_for_cart_item` and `traced_path_for` are thin convenience
    wrappers; prefer this helper directly when you're doing per-entity
    lookups within a single request (e.g. the web server iterating
    over a mosaic's targets).

    Returns `[]` if the entity isn't in context, the pipeline isn't
    registered, or the trace itself fails.

    **Snapshot-first (Ray-free):** if `compute_graph._store_tap_snapshot`
    persisted this exact `(source, stable, content, pipeline)` at graph-build
    time — which it does for every cart entry — return that directly (a
    `context.get`). A MISS means the object×pipeline was never built:
    `recompute=False` (the viewer / catalog) returns `[]`; `recompute=True`
    (sweep / archive on the head driver) recomputes the pure trace inline.
    """
    snapshot = _read_tap_snapshot(item)
    if snapshot is not None:
        return snapshot

    # Leaf has no snapshot of its own — but a cart-item ANCESTOR named in the URL
    # hierarchy does, and its shared-graph trace already holds this leaf's taps (a
    # target's snapshot carries every child stack's `Master` tap). Read the
    # ancestor's snapshot (a 0.1ms `context.get`) and return it whole; the
    # caller's `_routes_to_item` filter scopes it to the leaf. Only ONE ancestor
    # is the cart item, so the first hit is authoritative. The URL says which
    # snapshot holds the data — no scan, no index (feedback_avoid_derived_state).
    _, _, pipeline = item
    for anc_type, anc_id in ancestors:
        anc_snapshot = _read_tap_snapshot((anc_type, anc_id, pipeline))
        if anc_snapshot is not None:
            return anc_snapshot

    if not recompute:
        # Read-only caller (viewer / catalog): a snapshot miss (own AND every
        # named ancestor) means this object×pipeline was never built — nothing on
        # disk to show. Never trace on an HTTP request; just report "no taps".
        return []

    # Genuine on-demand miss on the head driver (sweep / archive of an item not
    # in the current graph). The trace is a PURE function — recompute it inline,
    # in-process. No @node, no Ray dispatch, no second diskcache to keep
    # canonical, no metadata_ref (that only existed to key the old @node).
    #
    # `strict=True` (sweep / archive) must NOT let a trace FAILURE degrade to an
    # empty result — that would drop the item from the live set and its on-disk
    # outputs would be swept as orphans. Each failure mode raises
    # TraceResolutionError instead. Non-strict callers (the viewer) keep the
    # tolerant `[]` so a broken product shows as "nothing" rather than 500ing.
    fn, entity, config = _resolve_pipeline_call(item)
    if fn is None:
        if strict:
            raise TraceResolutionError(
                f"cannot resolve pipeline/entity for cart item {item!r} — "
                "entity absent from context or pipeline unregistered"
            )
        return []
    from dependency_graph.tracing import trace
    try:
        graph = trace(fn, entity, config)
    except Exception as e:
        if strict:
            raise TraceResolutionError(
                f"tracing cart item {item!r} raised {type(e).__name__}: {e}"
            ) from e
        return []
    if graph is None:
        if strict:
            raise TraceResolutionError(
                f"tracing cart item {item!r} produced no graph"
            )
        return []
    return _walk_and_resolve_filenames_tagged(graph)


def traced_path_for(
    item: tuple[str, str, str],
    op_name: str,
    **identity_kwargs: str,
) -> "Path | None":
    """Find the single path produced by `op_name` for a specific entity
    in `item`'s traced pipeline.

    `identity_kwargs` are stable-hash values for the op's identity
    fields — e.g. `traced_path_for(item, "stack_files",
    stack=active_stack.stable_hash)`.

    Returns the first matching path, or None on miss. Convenience
    wrapper for one-shot lookups; if you're doing multiple lookups
    from the same cart item, call `traced_paths_for(item)` once and
    iterate.
    """
    for e in traced_paths_for(item):
        if e["op_name"] != op_name:
            continue
        if all(e["identity"].get(k) == v for k, v in identity_kwargs.items()):
            return Path(e["path"])
    return None


def _routes_to_item(entry: dict, object_type: str, object_id: str) -> bool:
    """Whether a traced tap belongs to the queried entity. A pipeline's trace
    contains taps routing to OTHER entities (a mosaic's trace has per-panel
    `Master` taps routing to stacks, per-target `Channel` taps routing to
    targets); only those routing to `(object_type, object_id)` belong on this
    entity's page.

    Match BOTH type and id: first-subframe-based `stable_hash` collides across
    a single-panel mosaic / its target / its stack, so id alone is ambiguous.
    `object_type` is the URL/cart form (`calibration_stack`); the tap carries
    the routing entity's class name (`CalibrationStack`)."""
    cls = entry.get("object_type")
    return (
        entry.get("object_id") == object_id
        and bool(cls)
        and object_type.replace("_", "").lower() == cls.lower()
    )


def pipeline_source_version(fn) -> str:
    """The cache-key version component for a pipeline's trace + tap snapshot:
    the pipeline function's own source hash combined with the DEPLOYED CODE
    VERSION (the frozen cluster SHA).

    `hash_source_code(fn)` alone covers only the pipeline body — so editing an
    op or helper the pipeline *calls* (e.g. `post_process_stack_oss`) would NOT
    move this hash, and the cached trace/tap-snapshot would keep the op's old
    call_hash (hence old output path), silently defeating the
    edit-invalidates-cache contract. Folding in `deployed_code_version()` means
    ANY code change re-derives the trace; an unrelated commit re-traces (cheap
    ~25 s) but rebuilds nothing (all call_hashes unchanged → cache hits).

    Both the writer (`compute_graph._store_tap_snapshot`, `run_pipeline`) and
    the readers (`_snapshot_key`, `build_main_processing_graph`) MUST route
    through this one function so their keys always agree. In dev/tests where no
    frozen SHA exists, `deployed_code_version()` is "" and this reduces to the
    previous pipeline-source-only hash — identical behavior."""
    from hashing import hash_source_code, hash_string
    from state.code_version import deployed_code_version
    return hash_string(hash_source_code(fn) + deployed_code_version())


def _snapshot_key(item: tuple[str, str, str]) -> "tuple | None":
    """The content-addressed `tap_snapshots` key for a cart item:
    `(pipeline_source_version(pipeline_fn), entity.stable_hash,
    entity.content_hash, pipeline)` — the SAME key
    `compute_graph._store_tap_snapshot` writes. Resolves the live entity from
    the (local) context and versions the pipeline source (+ code version).
    Returns None when the entity isn't in context or the pipeline isn't
    registered."""
    from pipelines.registry import get_pipeline
    from products.product import object_class

    object_type, object_id, pipeline = item
    type_to_dict = {
        "stack": context.stacks or {},
        "target": context.targets or {},
        "mosaic": context.mosaics or {},
    }
    cls = object_class(object_type)
    if cls is None:
        return None
    # Lowercase: the tap URL grammar sends PascalCase class names; the
    # context dicts key lowercase (see products.object_class).
    entity_ref = type_to_dict[str(object_type).lower()].get(object_id)
    if entity_ref is None:
        return None
    entity = entity_ref.get() if hasattr(entity_ref, "get") else entity_ref
    try:
        fn = get_pipeline(pipeline, processes=cls)._function
    except KeyError:
        return None
    return (pipeline_source_version(fn), entity.stable_hash, entity.content_hash, pipeline)


def _read_tap_snapshot(item: tuple[str, str, str]) -> "list[dict] | None":
    """Read the content-addressed snapshot persisted at graph-build time by
    `compute_graph._store_tap_snapshot`. **Ray-free** — resolve the live object
    from the (local) context, hash the pipeline source, build the SAME
    `(source_hash, stable_hash, content_hash, pipeline)` key, read the diskcache.
    No trace, no `ray.put`, no `.remote()`.

    Returns the stored entry list, or **None** when no snapshot matches — the
    pipeline isn't in the current graph, or the code / object content changed
    and it hasn't been re-traced yet. `traced_paths_for` treats None as "fall
    through to the on-demand Ray trace"; the web workers never have one, so for
    them a miss simply yields no taps (never a stale answer — old-key snapshots
    are harmless orphans, never read)."""
    key = _snapshot_key(item)
    if key is None:
        return None
    snap = (context.get("tap_snapshots") or {}).get(key)
    if snap is None:
        return None
    return (snap.get() if hasattr(snap, "get") else snap) or []


def invalidate_tap_snapshot(item: tuple[str, str, str]) -> bool:
    """Drop the build-time tap snapshot for `item` so the next compute-graph
    build re-walks and re-stores fresh resolved paths.

    `_store_tap_snapshot` keys only on the PIPELINE function's source hash (not
    the ops it calls) and early-returns on an existing entry. So when an op's
    source changes — its call_hash, hence output path, moves — the snapshot
    keeps the stale path and the dashboard resolves the wrong (or deleted) file,
    with nothing triggering a rebuild. A "discard results" action calls this so
    the post-discard `drain_and_restart` republishes the current paths. Returns
    True iff an entry was removed.

    Read-modify-write on the `tap_snapshots` dict; the immediately following
    rebuild is the source of truth, so a lost concurrent `or_equals` is benign."""
    key = _snapshot_key(item)
    if key is None:
        return False
    snaps = context.get("tap_snapshots") or {}
    if key not in snaps:
        return False
    new = dict(snaps)
    new.pop(key, None)
    context.set("tap_snapshots", new)
    return True


def tap_outputs_for(
    item: tuple[str, str, str],
    recompute: bool = True,
    ancestors: tuple[tuple[str, str], ...] = (),
) -> list[dict]:
    """The user-facing taps for `item`'s pipeline that belong to `item`'s
    entity: every `display=` tap routing to `(object_type, object_id)`. Sourced
    from the build-time snapshot (Ray-free via `traced_paths_for`); the dashboard
    groups by `display` + checks `os.path.exists` per filter. Replaces the old
    stateful PipelineOutputIndex `for_object` (keyed by object type+id likewise),
    but from a complete single-shot trace, not a two-phase recorder.

    `ancestors` lets a child object (a stack under a target) resolve its taps
    from the ancestor's snapshot; the `_routes_to_item` filter below then keeps
    only the taps that route to THIS leaf, so the shared ancestor trace is
    correctly scoped."""
    object_type, object_id, _pipeline = item
    return [
        e for e in traced_paths_for(item, recompute=recompute, ancestors=ancestors)
        if e.get("display") is not None
        and _routes_to_item(e, object_type, object_id)
    ]


def tap_path_for(
    item: tuple[str, str, str], display: str, filter: str | None,
    ancestors: tuple[tuple[str, str], ...] = (),
) -> "Path | None":
    """The on-disk primary path of one tap `(display, filter)` belonging to
    `item`'s entity, or None on miss. From the build-time snapshot (Ray-free);
    the file may or may not exist yet — callers check `os.path.exists`.

    `ancestors` (a child's cart-item chain) lets a leaf without its own snapshot
    resolve from the ancestor's; `_routes_to_item` scopes the shared trace."""
    object_type, object_id, _pipeline = item
    f = str(filter)
    # `recompute=False` to match this function's contract ("From the build-time
    # snapshot (Ray-free)") — it was omitted, so the DEFAULT True re-executed the
    # pipeline inside whatever process asked, which for the preview/tile routes
    # is a web worker. Same defect, same cost as the pipeline_outputs endpoint:
    # one `PixInsight Master` thumbnail measured 32 s. `catalog.products_for`
    # already passes it; this is the call site that didn't.
    for e in traced_paths_for(item, recompute=False, ancestors=ancestors):
        if (
            e.get("display") == display
            and str(e.get("filter")) == f
            and _routes_to_item(e, object_type, object_id)
        ):
            return Path(e["path"])
    return None


def _resolve_pipeline_call(
    item: tuple[str, str, str],
) -> tuple[Any, Any, Any]:
    """Return `(pipeline_body_fn, entity, ProcessingConfig)` ready to
    trace, or `(None, None, None)` when the cart entry is unresolvable.

    Bypasses the `@pipeline` runner so we don't trigger
    `set_workflow_key` side effects at sweep time.
    """
    object_type, object_id, pipeline_name = item

    from pipelines.registry import get_pipeline
    from products.product import object_class
    from cache.context_ref import ContextRef
    from decisions.settings import ProcessingConfig

    type_to_dict = {
        "stack": context.stacks or {},
        "target": context.targets or {},
        "mosaic": context.mosaics or {},
    }

    cls = object_class(object_type)
    if cls is None:
        return None, None, None

    # Lowercase: the tap URL grammar sends PascalCase class names; the
    # context dicts key lowercase (see products.object_class).
    entity_ref = type_to_dict[str(object_type).lower()].get(object_id)
    if entity_ref is None:
        return None, None, None
    entity = entity_ref.get() if hasattr(entity_ref, "get") else entity_ref

    try:
        runner = get_pipeline(pipeline_name, processes=cls)
    except KeyError:
        return None, None, None
    fn = runner._function

    # Mirror build_main_processing_graph's config construction so the
    # trace's call_hashes match the live graph's. `decisions` is the
    # full overrides set here; narrow_stack_config filters it to per-
    # stack relevant entries inside stack_files's narrow hook.
    settings = context.settings
    cal_stacks_ref = ContextRef.store_large(context.calibration_stacks or {})
    config = ProcessingConfig(
        calibration_stacks=cal_stacks_ref,
        development_settings=settings.development_settings,
        min_frames_in_stack=settings.clustering_settings.min_frames_in_stack,
        decisions=settings.decision_overrides,
    )
    return fn, entity, config


def resolve_output_paths(graph) -> dict:
    """Node -> absolute expected output path for every filename_fn node.

    The pure topological hash-propagation core shared with
    `_walk_and_resolve_filenames_tagged`: it computes each node's
    filename by threading `_TracedOutput` proxies (dep call-hashes)
    through downstream `filename_fn` calls. Deterministic function of
    the graph STRUCTURE (`Node.args` + `task.filename_fn` + propagated
    dep hashes) — it never reads runtime results, so it works for
    never-run nodes (the workflow-graph UI bakes these paths per node
    to check on-disk existence). Insertion order is topological. Never
    raises: returns `{}` (or a partial map) on failure.
    """
    try:
        order = _topological_order(graph)
    except Exception:
        return {}
    node_hash: dict[Any, str] = {}
    paths: dict[Any, str] = {}
    for n in order:
        if hasattr(n.task, "filename_fn"):
            try:
                args = _resolve_args(n, graph, node_hash)
                path = n.task.filename_fn(*args)
            except (AttributeError, KeyError, TypeError):
                # Missing upstream proxy or unhashable arg — skip the
                # subtree from here (downstream nodes that depend on us
                # will silently lack a hash for us).
                continue
            # Filename is `<output_dir>/.../<call_hash>.<ext>`; the
            # stem is exactly the call_hash.
            node_hash[n] = Path(path).stem
            paths[n] = str(path)
        else:
            # A PRESENTED subgraph builder (builder ≡ terminal,
            # `present_terminal`): its stamped path IS its terminal's, so
            # the workflow UI's on-disk checks and downstream filename_fn
            # args fold the deliverable's identity. Must win over dep
            # inheritance below, which would leak the SOURCE input's hash
            # into every downstream key.
            presented = getattr(n, "_presents_path", None)
            if presented is not None:
                node_hash[n] = Path(presented).stem
                paths[n] = str(presented)
                continue
            # Pass-through propagation for helpers without filename_fn.
            deps = list(graph.get(n, []) or [])
            inherited = next(
                (node_hash[d] for d in deps if d in node_hash), None,
            )
            if inherited is not None:
                node_hash[n] = inherited
            else:
                # Leaf-level resolver (best-effort). E.g.
                # `get_stack_subframe_image(stack)` returns a raw frame
                # path; its content_hash lives on the corresponding
                # ImageMetadata in `stack.all_metadata()`.
                leaf_hash = _resolve_leaf_content_hash(n)
                if leaf_hash is not None:
                    node_hash[n] = leaf_hash
    return paths


def _walk_and_resolve_filenames_tagged(graph) -> list[dict]:
    """`resolve_output_paths` + user-facing tap/identity metadata.

    Returns one dict per resolved filename:
        {
            "path": str,           # absolute on-disk path
            "op_name": str,        # producer's __name__ (e.g. "stack_files")
            "identity": dict,      # {field_name: stable_hash_of_arg}
            "is_final": bool,      # producer's `_final` attribute
        }
    Identity is what makes `traced_path_for(item, "stack_files",
    stack=H)`-style lookups possible: callers can pick out a specific
    (op, entity) combination from the trace without rehashing or
    re-tracing.
    """
    # User-facing tap metadata per Node, so a single traced-paths result serves
    # the dashboard/tiles (which key on the tap's routing entity + display +
    # filter) as well as sweep/archive (op_name+identity). The routing entity
    # (object_type/object_id) matters: a mosaic's trace contains per-panel
    # `Master` taps that route to STACKS and per-target `Channel` taps that
    # route to TARGETS — those belong on the stack/target views, not the
    # mosaic's. Non-display intermediates (e.g. place_image) aren't taps → all
    # four stay None.
    from compute_graph.compute_graph import extract_pipeline_taps
    taps_by_node: dict[Any, tuple] = {
        node: (object_type, object_id, display, filter_value)
        for node, (object_type, object_id, display, filter_value)
        in extract_pipeline_taps(graph)
    }

    out: list[dict] = []
    for n, path in resolve_output_paths(graph).items():
        tap_object_type, tap_object_id, display, filter_value = (
            taps_by_node.get(n, (None, None, None, None))
        )
        out.append({
            "path": path,
            "op_name": _op_name_for_node(n),
            "identity": _identity_for_node(n),
            # Node-level attr first: a presented builder carries its
            # TERMINAL's finality (see `present_terminal`).
            "is_final": bool(getattr(n, "_final", False)
                             or getattr(n.task, "_final", False)),
            "display": display,
            "filter": filter_value,
            "object_type": tap_object_type,
            "object_id": tap_object_id,
        })
    return out


def _op_name_for_node(n) -> str:
    # A presented builder answers with its TERMINAL's op name, so
    # `traced_path_for(item, "merge_images_oss", ...)`-style lookups
    # (archive, sweep) keep working across the subgraph boundary.
    presented = getattr(n, "_op_name", None)
    if presented:
        return presented
    fn = getattr(n.task, "_function", n.task)
    return getattr(fn, "__name__", "") or getattr(n.task, "__name__", "")


def _identity_for_node(n) -> dict:
    """Extract `{identity_field: stable_hash}` for the first applicable
    identity registered on `n.task`. Returns `{}` when no identity can
    be resolved (no _identities, or args don't bind cleanly).

    Each identity_cls's field names are treated as positional-parameter
    names on the producer; values come from `n.args` by parameter position.
    `stable_hash` is preferred over the raw object so the cached trace
    result stays small and easy to compare.
    """
    import dataclasses
    task = n.task
    fn = getattr(task, "_function", task)
    identities = getattr(task, "_identities", None) or []
    if not identities:
        return {}
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return {}
    params = list(sig.parameters.keys())
    arg_map = {
        params[i]: n.args[i]
        for i in range(min(len(n.args), len(params)))
    }
    # First identity that can bind wins. Ops registered for multiple
    # identities (stack_files for CalibrationStackId + StackId) all
    # share the same arg layout — the field names are the same — so
    # either binds equivalently.
    for identity_cls in identities:
        id_fields = [f.name for f in dataclasses.fields(identity_cls)]
        try:
            vals = {name: arg_map[name] for name in id_fields}
        except KeyError:
            continue
        return {
            k: (getattr(v, "stable_hash", None) or str(v))
            for k, v in vals.items()
        }
    return {}


def _find_dep_entity(dep, graph):
    """Walk `dep.args` + ancestor deps for the first value with
    `stable_hash`. Used to populate `_TracedOutput.entity` so polymorphic
    filename_fn dispatchers can isinstance-route through the proxy."""
    from dependency_graph.classes import Node
    for a in getattr(dep, "args", ()):
        if hasattr(a, "stable_hash"):
            return a
    for d in graph.get(dep, []) or ():
        if isinstance(d, Node):
            found = _find_dep_entity(d, graph)
            if found is not None:
                return found
    return None


def _resolve_args(
    node, graph, node_hash: dict[Any, str],
) -> list[Any]:
    """Build the argument list for `node.task.filename_fn`, mirroring
    `GraphRuntime.make_fn_args`: literal `node.args` followed by
    dep results. Dep results are replaced with `_TracedOutput` proxies
    carrying the dep's call_hash (or the propagated content_hash for
    pass-through helpers) plus the dep's primary entity for dispatcher
    routing."""
    sig = inspect.signature(node.task._function)
    args = list(node.args)
    if len(sig.parameters) > len(node.args):
        for dep in graph.get(node, []) or []:
            if dep in node_hash:
                args.append(_TracedOutput(
                    content_hash=node_hash[dep],
                    entity=_find_dep_entity(dep, graph),
                ))
            else:
                # Unresolved dep — append a sentinel that will cause
                # filename_fn to raise (caught upstream → skip subtree).
                args.append(None)
    return args


def _resolve_leaf_content_hash(node) -> str | None:
    """Try to resolve a leaf helper's effective content_hash without
    actually running heavy code. Handles `get_stack_subframe_image`
    specifically — its arg is a `Stack`, and the picked frame's
    content_hash is already on the stack's per-frame ImageMetadata."""
    fn_name = getattr(node.task._function, "__name__", "")
    if fn_name == "get_stack_subframe_image" and node.args:
        stack = node.args[0]
        try:
            from stacks.compute_graph import get_light_frames_sorted_by_stars
            best = get_light_frames_sorted_by_stars(stack)[0]
            meta = stack.all_metadata()[best]
            return meta.content_hash
        except Exception:
            return None
    return None


def _topological_order(graph) -> list:
    """Stable topological sort of `graph` (a dict {Node: [dep, ...]}).
    Same semantics as `GraphRuntime`'s ready/done scheduling, just
    flattened for offline walking."""
    sorter: TopologicalSorter = TopologicalSorter()
    for n, deps in graph.items():
        sorter.add(n, *(deps or ()))
    return list(sorter.static_order())
