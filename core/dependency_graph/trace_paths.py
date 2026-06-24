"""Trace a cart item's pipeline and return every expected output path.

Under the pure-Merkle `content_hash := producer.call_hash` convention
(stamped into file headers by `@cached.remote`), filenames are deterministic
and statically computable from the trace alone — no file I/O at trace time
beyond ImageMetadata already in context.

The trace itself can be expensive (real measurement on a "PixInsight
Downscaled" mosaic with 730 nodes: ~17 s — dominated by
`resolve_calibration`'s filesystem-walking calibration matching). Since
the trace result is a deterministic pure function of (pipeline source,
entity, ProcessingConfig, metadata), it's cached as a `@node`:
`_traced_paths`. First call pays the cost; every subsequent call with the
same inputs is a sub-ms diskcache hit, durably persisted across restarts.

**Developer note on cache invalidation.** The trace cache key invalidates
when ANY of these change:
  - pipeline body source (`pipeline_source_hash` arg)
  - entity (via stable_hash)
  - ProcessingConfig (decisions, dev settings, cal_stacks ref)
  - metadata_ref content_hash (subframe set + external master set)
It does NOT invalidate when helper functions change (`resolve_calibration`,
`find_external_master`, `_stack_files_narrow`, etc.) — consistent with
the codebase-wide policy that `@node`'s fn_hash covers only the wrapped
function, not its transitive closure. When you edit one of those helpers
and need the trace to re-fire, bust the relevant cache entries explicitly.

Used by `scripts.sweep_orphans` to compute the live set and by
`state.archive` to populate archive entries at cart→archive transition.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Any

from context import local_context as context
from decorators import node


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
) -> list[Path]:
    """Trace `item`'s pipeline against the current context and return
    every `filename_fn`-resolvable output path.

    `item` is the canonical cart-entry tuple `(object_type, object_id,
    pipeline_name)`. Returns `[]` if the entity isn't in context (stale
    cart entry) or the pipeline isn't registered.

    With `finals_only=True`, restricts to outputs of ops decorated
    `@node(... final=True)` — used by archive transition with
    `preserve="finals"`.

    Dispatches via the cached `_traced_paths` `@node`: first call per
    (pipeline, entity, config, metadata, source) pays the trace cost
    (~17 s for a large mosaic); every subsequent call with identical
    inputs is a diskcache hit.
    """
    return [Path(e["path"]) for e in traced_paths_for(item)
            if not finals_only or e["is_final"]]


def traced_paths_for(item: tuple[str, str, str]) -> list[dict]:
    """Return the raw trace result for `item`: one dict per
    filename_fn-resolvable node, with keys `path`, `op_name`,
    `identity`, `is_final`.

    One Ray round-trip per call. Use this when you need to look up
    multiple paths from the same cart item — hoist it out of any loop,
    iterate the returned list in pure Python.

    `paths_for_cart_item` and `traced_path_for` are thin convenience
    wrappers; prefer this helper directly when you're doing per-entity
    lookups within a single request (e.g. the web server iterating
    over a mosaic's targets).

    Returns `[]` if the entity isn't in context, the pipeline isn't
    registered, or the trace itself fails.
    """
    import ray
    from hashing import hash_source_code
    from metadata.image_metadata import load_metadata

    fn, entity, config = _resolve_pipeline_call(item)
    if fn is None:
        return []

    # Prefer the published ContextRef (cheap diskcache read or one
    # actor RPC; the ref itself is small). Fall back to a fresh scan
    # only when nothing has published yet (first ever call before any
    # regroup cycle has run). load_metadata is ~20-30 s on a populated
    # cache, and we need its result on every call here to form the
    # trace @node's cache key — without the published ref the cache
    # might be hot but the web server still pays the full scan cost.
    metadata_ref = context.get("metadata_ref")
    if metadata_ref is None:
        metadata_ref = ray.get(load_metadata.remote())

    pipeline_source_hash = hash_source_code(fn)
    _, _, pipeline_name = item

    try:
        return ray.get(_traced_paths.remote(
            pipeline_name, entity, config, metadata_ref, pipeline_source_hash,
        ))
    except Exception:
        return []


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


def tap_outputs_for(item: tuple[str, str, str]) -> list[dict]:
    """The user-facing taps for `item`'s pipeline: every traced output that
    carries a `display=` name. One cached trace; the web dashboard groups
    these by `display` and checks `os.path.exists` per filter. Replaces the
    old stateful PipelineOutputIndex `for_object`."""
    return [e for e in traced_paths_for(item) if e.get("display") is not None]


def tap_path_for(
    item: tuple[str, str, str], display: str, filter: str | None,
) -> "Path | None":
    """The on-disk primary path of one tap `(display, filter)` in `item`'s
    traced pipeline, or None on miss. Pure trace prediction — the file may or
    may not exist yet; callers check `os.path.exists` as needed."""
    f = str(filter)
    for e in traced_paths_for(item):
        if e.get("display") == display and str(e.get("filter")) == f:
            return Path(e["path"])
    return None


# Head-pinned: like trace_pipeline, this is orchestration whose @node
# wrapper memoizes a file-less result (list[dict]) into the local
# diskcache. Only the head has the `diskcache` resource, so requesting a
# sliver keeps it on the head's canonical cache instead of SPREAD-ing onto
# a worker's local disk (the silent-write-to-worker-disk bug).
@node(resources={"diskcache": 0.001})
def _traced_paths(
    pipeline_name: str,
    entity,
    config,
    metadata_ref,
    pipeline_source_hash: str,
) -> list[dict]:
    """Cacheable core of `paths_for_cart_item` and `traced_path_for`.

    Returns one dict per filename_fn-resolvable node:
    `{"path", "op_name", "identity", "is_final"}`. Tagging every
    relevant axis in one cache entry means a single trace serves all
    consumers (sweep_orphans wants finals_only; the web server's
    metadata route wants identity-based lookup; archive wants finals).

    `pipeline_source_hash` is an invalidation key only — looked up at
    runtime via `pipeline_name` (the source hash is captured here purely
    to participate in `canonical_call_hash`). `metadata_ref` similarly:
    its content_hash enters the cache key; the trace body doesn't
    consume it directly (the pipeline's downstream tasks read it via
    `config.calibration_stacks` and `context` lookups).
    """
    from dependency_graph.tracing import trace

    fn = _lookup_pipeline_fn(pipeline_name, entity)
    if fn is None:
        return []
    try:
        graph = trace(fn, entity, config)
    except Exception:
        return []
    if graph is None:
        return []
    return _walk_and_resolve_filenames_tagged(graph)


def _lookup_pipeline_fn(pipeline_name: str, entity):
    from pipelines.registry import get_pipeline
    try:
        return get_pipeline(pipeline_name, processes=type(entity))._function
    except KeyError:
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
    from stacks.composite_stack import Stack
    from targets.types import Target
    from mosaics.types import Mosaic
    from cache.context_ref import ContextRef
    from decisions.settings import ProcessingConfig

    type_to_class = {"stack": Stack, "target": Target, "mosaic": Mosaic}
    type_to_dict = {
        "stack": context.stacks or {},
        "target": context.targets or {},
        "mosaic": context.mosaics or {},
    }

    cls = type_to_class.get(object_type)
    if cls is None:
        return None, None, None

    entity_ref = type_to_dict[object_type].get(object_id)
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


def _walk_and_resolve_filenames_tagged(graph) -> list[dict]:
    """Topologically walk `graph`, computing each filename_fn-bearing
    node's filename and threading `_TracedOutput` proxies through to
    downstream nodes' filename_fn calls.

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

    Non-filename_fn nodes (`compute_image_metadata`,
    `get_stack_subframe_image`, helpers like `calibrate`) don't write
    files but DO appear as deps of filename_fn nodes. We propagate a
    content_hash through them: a node with a single dep inherits that
    dep's content_hash (compute_image_metadata is a pass-through —
    `content_hash(ImageMetadata) == content_hash(source file)` under
    pure Merkle). A leaf without a dep tries to resolve from context.
    """
    order = _topological_order(graph)
    node_hash: dict[Any, str] = {}
    out: list[dict] = []

    # User-facing tap metadata (display name + filter) per Node, so a single
    # traced-paths result serves the dashboard/tiles (which key on
    # display+filter) as well as sweep/archive (op_name+identity). Non-display
    # intermediates (e.g. place_image) aren't taps → display/filter stay None.
    from compute_graph.compute_graph import extract_pipeline_taps
    taps_by_node: dict[Any, tuple[str, "str | None"]] = {
        node: (display, filter_value)
        for node, (_ot, _oid, display, filter_value) in extract_pipeline_taps(graph)
    }

    for n in order:
        if hasattr(n.task, "filename_fn"):
            args = _resolve_args(n, graph, node_hash)
            try:
                path = n.task.filename_fn(*args)
            except (AttributeError, KeyError, TypeError):
                # Missing upstream proxy or unhashable arg — skip the
                # subtree from here (downstream nodes that depend on us
                # will silently lack a hash for us).
                continue
            # Filename is `<output_dir>/.../<call_hash>.<ext>`. The
            # stem is exactly the call_hash.
            node_hash[n] = Path(path).stem
            display, filter_value = taps_by_node.get(n, (None, None))
            out.append({
                "path": str(path),
                "op_name": _op_name_for_node(n),
                "identity": _identity_for_node(n),
                "is_final": bool(getattr(n.task, "_final", False)),
                "display": display,
                "filter": filter_value,
            })
        else:
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

    return out


def _op_name_for_node(n) -> str:
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
