import ray
import structlog
from typing import Any
from beartype import beartype

from context import context
from cache.context_ref import ContextRef
from metadata.classes import DefectsDict, MetadataDict
from metadata.image_metadata import scatter_compute_image_metadata
from clustering import cluster_by_image_center
from calibration import (
    identify_calibration_stacks,
    associate_calibration_stacks,
)
from previews.stack_thumbnails import scatter_stack_thumbnails
from mosaics.identification import identify_mosaics
from stacks.compute_graph import (
    process_stacks_for_ui,
    process_calibration_stacks_for_ui,
)
from targets.compute_graph import process_targets_for_ui
from targets.identification import identify_targets
from targets.types import TargetsDict
from mosaics.compute_graph import process_mosaics_for_ui
from mosaics.types import MosaicsDict
from dependency_graph import Graph, Node
from stacks.identification import identify_stacks
from stacks.composite_stack import RunsDict, StacksDict, CalibrationStacksDict
from decorators import node
from decisions.compute_graph import (
    scatter_customize_targets,
)
from decisions.settings import ClusteringSettings, DevelopmentSettings
from decisions.decisions import decisions_for_objects
from compute_graph.objects import Objects

log = structlog.get_logger()


@ray.remote(num_cpus=0.1)
@beartype
def store_calibration_stacks(
    calibration_stacks: CalibrationStacksDict,
) -> CalibrationStacksDict:
    context.calibration_stacks = calibration_stacks
    log.info("Stored calibration stacks", count=len(calibration_stacks))
    return calibration_stacks


@ray.remote(num_cpus=0.1)
@beartype
def store_stacks(stacks: dict[str, Any]) -> dict[str, Any]:
    assert None not in stacks.values()
    # stacks values are already ContextRefs (from gather_enriched_stacks)
    result = StacksDict(stacks)
    context.stacks = result
    log.info("Stored stacks", count=len(stacks))
    return result


@ray.remote(num_cpus=0.1)
@beartype
def store_targets(targets: dict[str, Any]) -> dict[str, Any]:
    assert None not in targets.values(), "ContextRef itself is None — identify_targets bug"
    # Validate that every ContextRef's underlying value is reachable.
    # We've hit a recurring bug where 2-4 target ContextRefs in the
    # durable cache go missing between writes — `ref.get()` returns
    # None. Surfacing that at write time, with the offending hashes,
    # so the failure is visible at the source instead of propagating
    # into `identify_mosaics` (which crashes on `target.center[0]`).
    missing = [h for h, ref in targets.items() if ref.get() is None]
    if missing:
        raise RuntimeError(
            f"store_targets: {len(missing)} target refs have no underlying "
            f"value in the durable cache: {missing[:8]}. This is the bug "
            f"that breaks identify_mosaics — investigate ContextRef.store / "
            f"diskcache write path."
        )
    result = TargetsDict(targets)
    context.targets = result
    log.info("Stored targets", count=len(targets))
    return result


@ray.remote(num_cpus=0.1)
@beartype
def store_mosaics(mosaics: MosaicsDict) -> MosaicsDict:
    assert None not in mosaics.values()
    # mosaics values are already ContextRefs (from identify_mosaics)
    context.mosaics = mosaics
    log.info("Stored mosaics", count=len(mosaics))
    return mosaics


@ray.remote(num_cpus=0.1)
def store_ui_stacks(ui_stacks: dict) -> None:
    context.ui_stacks = ui_stacks


@ray.remote(num_cpus=0.1)
def store_ui_calibration_stacks(ui_cal_stacks: dict) -> None:
    context.ui_calibration_stacks = ui_cal_stacks


@ray.remote(num_cpus=0.1)
def store_ui_targets(ui_targets: dict) -> None:
    context.ui_targets = ui_targets


@ray.remote(num_cpus=0.1)
def store_ui_mosaics(ui_mosaics: dict) -> None:
    context.ui_mosaics = ui_mosaics


@ray.remote(num_cpus=0.1)
@beartype
def combine_dicts(*ancestor_results: dict[Any, Any]) -> dict[Any, Any]:
    assert isinstance(ancestor_results, tuple)
    for d in ancestor_results:
        assert None not in d.values(), f"### combine_dicts: None in dict {d}"
    result = {key: value for d in ancestor_results for key, value in d.items()}
    assert None not in result.values()
    return result


@ray.remote(num_cpus=0.1)
def scatter_spatial_groups(
    metadata_ref: ContextRef[MetadataDict],
    clustering_settings: ClusteringSettings,
    decision_overrides,
    *ancestor_results,
) -> Graph:
    from cache.context_ref import ContextRef

    # Two ancestor results in non-deterministic order:
    #   - cluster_by_image_center returns list[tuple[str, ...]]
    #   - identify_camera_fingerprints returns ContextRef[FingerprintMap]
    # Discriminate by type.
    spatial_groups = next(
        r for r in ancestor_results if not isinstance(r, ContextRef)
    )
    fingerprint_map_ref = next(
        r for r in ancestor_results if isinstance(r, ContextRef)
    )

    graph = Graph({})

    identify_cal_stacks_task = (
        identify_calibration_stacks,
        metadata_ref,
        clustering_settings.max_run_gap,
        clustering_settings.min_frames_in_stack,
        fingerprint_map_ref,
    )

    # Cal-to-cal association runs through the same decision framework as light
    # calibration (`resolve_calibration`), so it takes the decision overrides.
    associate_cal_task = (associate_calibration_stacks, decision_overrides)
    graph[associate_cal_task] = {identify_cal_stacks_task}
    graph[store_calibration_stacks] = {associate_cal_task}
    graph[process_calibration_stacks_for_ui] = {associate_cal_task}
    graph[store_ui_calibration_stacks] = {process_calibration_stacks_for_ui}

    # ORDERED lists, not sets: these deps feed value-consuming nodes
    # (combine_dicts takes *ancestor_results), and dep order is part of the
    # cache key — canonicalize_graph now raises on unordered value-deps.
    stack_tasks = []
    target_tasks = []
    for i, g in enumerate(spatial_groups):
        stack_task = (
            identify_stacks,
            metadata_ref,
            clustering_settings.angular_tolerance,
            g,
            clustering_settings.min_frames_in_stack,
            clustering_settings.max_run_gap,
            fingerprint_map_ref,
        )

        target_task = Node(identify_targets, clustering_settings.angular_tolerance, name=str(i))
        customize_targets_task = Node(
            scatter_customize_targets, name=str(i)
        )

        # Each target identification depends on its own spatial group's stacks.
        graph[target_task] = {stack_task}
        graph[customize_targets_task] = {target_task}

        stack_tasks.append(stack_task)
        target_tasks.append(customize_targets_task)

    gather_stacks_task = Node(combine_dicts, name="combine_stacks")
    graph[gather_stacks_task] = stack_tasks
    graph[store_stacks] = {gather_stacks_task}
    process_stacks_task = (
        process_stacks_for_ui,
        decision_overrides,
        clustering_settings.min_frames_in_stack,
    )
    # Ordered: process_stacks_for_ui consumes both results positionally
    # (it unpacks by type, but the call HASH folds argument order).
    graph[process_stacks_task] = [store_stacks, store_calibration_stacks]
    graph[store_ui_stacks] = {process_stacks_task}

    gather_targets_task = Node(combine_dicts, name="combine_targets")
    graph[gather_targets_task] = target_tasks
    graph[store_targets] = {gather_targets_task}
    graph[process_targets_for_ui] = {gather_targets_task}
    graph[store_ui_targets] = {process_targets_for_ui}

    mosaics_task = Node(identify_mosaics, clustering_settings.angular_tolerance)
    graph[mosaics_task] = {gather_targets_task}
    graph[store_mosaics] = {mosaics_task}
    graph[process_mosaics_for_ui] = {mosaics_task}
    graph[store_ui_mosaics] = {process_mosaics_for_ui}

    # gather_objects is the sole terminal node — its result (Objects) flows
    # to downstream graph builders via the chain.
    graph[gather_objects] = {
        store_stacks, store_targets, store_mosaics, store_calibration_stacks,
        store_ui_stacks, store_ui_targets, store_ui_mosaics, store_ui_calibration_stacks,
    }

    return graph


@ray.remote(num_cpus=0.1)
@beartype
def build_regroup_graph(
    clustering_settings: ClusteringSettings,
    decision_overrides,
    metadata_ref: ContextRef[MetadataDict],
) -> Graph:
    # `scatter_compute_image_metadata`'s subgraph terminates in
    # `publish_library_refs`, which publishes BOTH `metadata_ref` and
    # `defects_ref` into context and returns `metadata_ref` as the
    # subgraph's final result. By the time this builder fires, both refs
    # are in context. `metadata_ref` flows in as our positional ancestor;
    # `defects_ref` is read from context.
    from stacks.identification import identify_camera_fingerprints

    defects_ref: ContextRef[DefectsDict] = context["defects_ref"]

    cluster_task = (
        cluster_by_image_center,
        metadata_ref,
        clustering_settings.overlap_fraction,
        clustering_settings.min_component_members,
    )
    fingerprint_task = (
        identify_camera_fingerprints,
        metadata_ref,
        defects_ref,
        clustering_settings.fingerprint_min_anchor_size,
    )
    scatter_task = (
        scatter_spatial_groups,
        metadata_ref,
        clustering_settings,
        decision_overrides,
    )

    return Graph(
        {
            cluster_task: {},
            fingerprint_task: {},
            # Ordered: scatter_spatial_groups consumes both results via
            # *ancestor_results, so dep order is part of its call hash.
            scatter_task: [cluster_task, fingerprint_task],
        }
    )


@ray.remote(num_cpus=0.1)
@beartype
def gather_objects() -> Objects:
    """Snapshot all identified objects from context after regrouping."""
    return Objects(
        stacks=context.stacks or StacksDict(),
        targets=context.targets or TargetsDict(),
        mosaics=context.mosaics or MosaicsDict({}),
        calibration_stacks=context.calibration_stacks or CalibrationStacksDict({}),
    )


@node
def preview_graph_done() -> None:
    pass


@node
def build_preview_graph(
    development_settings: DevelopmentSettings,
    min_frames_in_stack: int,
    code_version: str,
    objects: Objects,
) -> Graph:
    # `code_version` is a cache-key arg only (see build_main_processing_graph):
    # any code change re-derives this cached preview graph.
    from decisions.settings import ProcessingConfig
    from cache.context_ref import ContextRef

    cal_stacks_ref = ContextRef.store_large(objects.calibration_stacks)
    config = ProcessingConfig(
        calibration_stacks=cal_stacks_ref,
        development_settings=development_settings,
        min_frames_in_stack=min_frames_in_stack,
    )

    graph = Graph({})
    tasks = [(scatter_stack_thumbnails, cal_stacks_ref, objects.stacks)]

    # Subframe pipeline body resolves `mosaic.targets` against
    # `context.targets`; populate it here so each `run_pipeline`
    # task's trace has access to it.
    from context import local_context as context
    context.targets = objects.targets

    # Auto-preview: the 'Subframe' pipeline is dispatched over every
    # mosaic — semantically "every mosaic is implicitly in the cart
    # with the Subframe pipeline." Same per-pipeline sub-runtime
    # pattern as the cart-driven main-processing graph (see
    # `run_pipeline` above), so each Subframe trace's
    # `node_status_fn` / `graph_update_fn` fires inside its own
    # sub-runtime.
    #
    # Eagerly trace each Subframe pipeline here on the head-side @node
    # worker so the global ephemeral_files registry sees every consumer
    # before any Subframe sub-Graph dispatches. The trace is cheap
    # Python — no @node bodies run — and `run_pipeline` re-traces lazily
    # on the cluster with bit-identical Node identities.
    traced: list = []
    for mosaic_ref in objects.mosaics.values():
        mosaic = mosaic_ref.get()
        task = (run_pipeline, mosaic, config, "Subframe", "mosaic")
        graph[task] = set()
        tasks.append(task)
        traced.append(_eager_trace_pipeline("Subframe", "mosaic", mosaic, config))

    graph[preview_graph_done] = tasks
    graph.ephemeral_files = _build_ephemeral_registry(traced)
    return graph


@ray.remote(num_cpus=0.1)
@beartype
def run_pipeline_done() -> None:
    """Per-pipeline terminal sentinel. `GraphRuntime` asserts a
    sub-graph has exactly one node with no descendants; a real
    pipeline trace has many (per-filter sinks, register-tap-path
    nodes, etc.), so we funnel them all into this no-op so the
    sub-runtime has a single completion handle."""
    pass


def _eager_trace_pipeline(
    pipeline_name: str, type_tag: str, entity, cfg,
) -> "Graph | None":
    """Trace a pipeline on the head node (in-process, not via Ray).

    Same code path as the lazy trace inside `run_pipeline` — calls
    `pipeline_fn(entity, cfg)` from the registry and walks tap points.
    Used by `build_main_processing_graph` / `build_preview_graph` to
    pre-populate the global ephemeral_files registry before any
    consumer of an ephemeral file can dispatch. `run_pipeline` still
    re-traces lazily at runtime; the two traces are bit-identical
    (deterministic, same args, same context), and structural Node
    identity means the eager-trace's consumers match the lazy-trace's
    consumers exactly in the registry.

    Returns None if the pipeline body bails (e.g. MASTER-mode mosaic
    with missing panels) — callers should skip registry population
    for that entry.
    """
    from pipelines.registry import get_pipeline
    from products.product import object_class
    from dependency_graph.graph_runtime import GraphRuntime
    from dependency_graph.workflow_graph_adapter import store_workflow_graph

    pipeline_fn = get_pipeline(pipeline_name, processes=object_class(type_tag))
    graph = pipeline_fn(entity, cfg)
    if graph is None:
        return None
    # Topology-only snapshot so the viewer can render structure
    # before any node completes. Lives here, NOT in the @pipeline
    # runner, so dashboard tap-extraction traces don't pay for it.
    store_workflow_graph(
        (type_tag, entity.stable_hash, pipeline_name), GraphRuntime(graph),
    )
    return graph


def _version_supersede_map(pipeline_name: str, graph) -> dict:
    """`{node: (version_log_key, expected_path)}` for every display-tap node in
    `graph`. Attached to the sub-graph `run_pipeline` returns; the graph
    runtime consults it at node completion to delete the superseded (N-1)
    version's files THE MOMENT the new output lands — the deterministic,
    write-anchored half of graceful N/N-1 versioning (`version_log.supersede`).
    Pure walk (resolve_output_paths + extract_pipeline_taps), ~ms."""
    from dependency_graph.trace_paths import resolve_output_paths
    from products import version_log
    paths = resolve_output_paths(graph)
    return {
        node: (
            version_log.key(object_type, object_id, pipeline_name,
                            display, filter_value),
            paths[node],
        )
        for node, (object_type, object_id, display, filter_value)
        in extract_pipeline_taps(graph)
        if display is not None and node in paths
    }


def _store_tap_snapshot(pipeline_fn, pipeline_name: str, entity, graph) -> None:
    """Persist a cart entry's complete display-tap list so the Ray-free web
    server can read it without tracing.

    CONTENT-ADDRESSED key — `(pipeline_source_hash, stable_hash, content_hash,
    pipeline)`. A pipeline-code change (new source hash) or new images on the
    object (new content_hash) yields a NEW key, so the web UI — which computes
    the SAME key from the live object + pipeline source — always reads the
    current trace; stale snapshots are harmless orphans, never requested, aged
    out by LRU. One atomic write (no two-phase `record_path` to drift). The web
    worker filters the full list to the queried routing entity on read."""
    from cache.context_ref import ContextRef
    from dependency_graph.trace_paths import (
        _walk_and_resolve_filenames_tagged, pipeline_source_version,
    )

    # Compute the content-addressed key FIRST and skip the walk when a snapshot
    # for this exact (source, stable, content, pipeline) is already present AND
    # still resolves (not LRU-evicted). So `run_pipeline` on a restart with an
    # unchanged cart confirms presence in ~ms instead of re-walking — the walk
    # is cheap (~0.5 s) but this makes a warm restart free, and the expensive
    # part (the ~25 s trace) was already a `trace_pipeline` @node cache hit.
    #
    # NOTE: this stores ONLY the cart item's key — but that one snapshot holds
    # EVERY descendant's taps too (a target's snapshot contains its stacks'
    # `Master` taps; a mosaic's contains its targets' `Channel` taps), because
    # the shared graph produced them. A child view is surfaced by reading THIS
    # (the ancestor's) snapshot and filtering to the child via `_routes_to_item`
    # — the reader supplies the ancestor from the URL hierarchy. No separate
    # child key, no re-key, no closure to maintain.
    #
    # `pipeline_source_version` folds the deployed code SHA into the source
    # component so an op edit re-keys the snapshot (the reader `_snapshot_key`
    # uses the SAME helper, keeping writer/reader keys in lockstep).
    key = (
        pipeline_source_version(pipeline_fn._function),
        entity.stable_hash, entity.content_hash, pipeline_name,
    )
    existing = (context.get("tap_snapshots") or {}).get(key)
    if existing is not None and (
        existing.get() if hasattr(existing, "get") else existing
    ):
        return

    # The COMPLETE trace (every filename node, not just display-taps): the web
    # server reads it for the dashboard taps (display+filter), the mosaic
    # polygon overlays (stack_files by identity), and file opens — all Ray-free.
    entries = _walk_and_resolve_filenames_tagged(graph)
    if not entries:
        return
    ref = ContextRef.store(("tap_snapshots", str(key)), entries)
    context.or_equals("tap_snapshots", {key: ref})

    # Record each display product's new output in its N/N-1 version log. We only
    # reach here when the content-addressed snapshot changed (new source/content
    # hash), but the per-op call_hash dedup inside `record` means an unchanged
    # display tap (same call_hash as before) logs nothing — so a change that
    # doesn't move a given tap's output leaves its history untouched. Only
    # display taps participate (`display is not None`); non-display intermediates
    # are never shown, so they carry no version history.
    from products import version_log
    for e in entries:
        if e.get("display") is not None:
            version_log.record(
                version_log.key(e["object_type"], e["object_id"], pipeline_name,
                                e["display"], e["filter"]),
                e["path"],
            )


def _build_ephemeral_registry(traced_graphs):
    """Union the ephemeral_files contributions from every traced
    pipeline graph into a single registry. Returns the registry; the
    caller attaches it to the Graph it returns from a builder @node."""
    from dependency_graph.ephemeral_registry import EphemeralFiles
    registry = EphemeralFiles()
    for g in traced_graphs:
        if g is not None:
            registry.register_from_graph(g)
    return registry


@ray.remote(num_cpus=0.1)
def run_pipeline(entity, cfg, pipeline_name: str, type_tag: str) -> "Graph":
    """Reach for the cached trace, funnel terminals into the
    sub-runtime sentinel. Returns the Graph so the parent runtime
    spins up a child `GraphRuntime`.

    Trace is via the @node-cached `trace_pipeline` — body only re-runs
    when the pipeline source, entity, or cfg changes. The cached graph
    is pure data; tap paths are derived on demand by the dashboard from
    the same cached trace + disk (no runtime tap recording).

    Returning a Graph from a `@ray.remote` task is how `run_graph`
    spins up a child `GraphRuntime`. That sub-runtime reads the
    tracer's `node_status_fn` (= `store_node_status`) and
    `graph_update_fn` (= `partial(store_workflow_graph, pipeline_key)`)
    straight off the returned Graph — so per-pipeline live state
    updates fire naturally as nodes complete, without parent-level
    plumbing.
    """
    from pipelines.registry import get_pipeline
    from products.product import object_class
    from dependency_graph import Graph

    pipeline_fn = get_pipeline(pipeline_name, processes=object_class(type_tag))

    # Cached trace. The graph is freshly deserialized from diskcache
    # on each cache hit so in-place mutation downstream is safe. The version
    # arg folds in the deployed code SHA so editing ANY op the pipeline calls
    # (not just the pipeline body) re-derives the trace — otherwise the cached
    # graph keeps the op's old call_hash and no rebuild is triggered.
    from dependency_graph.trace_paths import pipeline_source_version
    source_hash = pipeline_source_version(pipeline_fn._function)
    graph = ray.get(trace_pipeline.remote(
        pipeline_name, source_hash, entity, cfg,
    ))
    if graph is None:
        # MASTER-mode mosaics with missing panels legitimately bail.
        # Surface an empty graph so the parent task's `node_completed`
        # still gets a Graph and closes out the cart entry cleanly.
        return Graph({})

    # Persist the tap snapshot for the Ray-free web server. Done HERE (not in
    # the @node-cached build_main_processing_graph, which is a cache-hit on an
    # unchanged cart and would never re-store on restart) because run_pipeline
    # runs on every `run_entire_graph`. The trace above is the cached
    # `trace_pipeline` (no 500 s re-trace), and `_store_tap_snapshot` skips the
    # walk entirely when the content-addressed snapshot already exists — so a
    # restart with an unchanged cart just confirms presence and moves on.
    _store_tap_snapshot(pipeline_fn, pipeline_name, entity, graph)

    # Deterministic N-1 supersession: the runtime deletes a display tap's
    # previous version at the completion of the node that writes the new one
    # (`version_log.supersede`). Attached every run — including unchanged
    # rebuilds, where the same-hash entry makes supersede a no-op.
    graph.version_supersede = _version_supersede_map(pipeline_name, graph)

    # GraphRuntime asserts a sub-graph has exactly one terminal. Funnel
    # every current terminal into a sentinel so the child sub-runtime
    # has a single completion handle.
    depended_on = {d for ds in graph.values() for d in ds}
    terminals = [n for n in graph if n not in depended_on]
    graph[run_pipeline_done] = terminals
    return graph


@ray.remote(num_cpus=0.1)
@beartype
def main_processing_graph_done() -> None:
    pass


def _find_identity_arg(graph, node):
    """Walk `node.args` + dep ancestors for the first value that has
    `stable_hash` (i.e. an identity-bearing Stack/Target/Mosaic).

    The tracer's convention is "literals first, deps trailing": deps
    get stripped from `Node.args` and stored on graph edges, so an op
    like `stack_files(stack, config, *cal_masters)` where `stack` is a
    dep ends up with `args = (config,)` and `deps = [stack_node, ...]`.
    The entity is somewhere down that dep chain — usually as the
    leading literal of the first dep — so we recurse.
    """
    from dependency_graph.classes import Node
    for a in node.args:
        if hasattr(a, "stable_hash"):
            return a
    for d in graph.get(node, []) or ():
        if isinstance(d, Node):
            found = _find_identity_arg(graph, d)
            if found is not None:
                return found
    return None


def extract_pipeline_taps(graph):
    """Pure walk of a traced `graph` yielding `(node, (object_type_name,
    object_id, display, filter))` per @makes_image tap.

    "Tap" = any Node whose task has a `_display` attribute (set by
    `@makes_image(display=...)`). Same-key collisions inside a single
    graph (a pipeline that calls `merge_images` twice on the same
    (mosaic, filter), say) get the `display` suffixed with ` 2`, ` 3`,
    …; the first occurrence keeps its bare display so single-merge
    pipelines don't show "Merged 1".

    No side effects, no graph mutation.
    """
    from dependency_graph.classes import Node
    from dependency_graph.node_introspection import filter_of

    if graph is None:
        return

    # Per-(type, id, display, filter) sequence count for disambiguation.
    seq: dict[tuple, int] = {}

    for n in list(graph.keys()):
        if not isinstance(n, Node):
            continue
        task = n.task
        fn = getattr(task, "_function", task)
        base_display = getattr(fn, "_display", None) or getattr(task, "_display", None)
        if base_display is None:
            continue
        # Ephemeral nodes (inside an `ephemeral_intermediates()` scope and not
        # `final`) are reclaimed after their consumers — not user-facing — so
        # they register no tap, even with a `display=`. `display=` "activates"
        # only when the node is kept (unwrapped or final).
        if getattr(n, "_ephemeral", False):
            continue
        # Find the entity: walks `n.args` for an instance with
        # `stable_hash`, then descends into upstream deps if needed.
        # Lets ops with `stack` as a trailing dep (e.g. `stack_files
        # (stack, config, *cal_masters)`) route taps the same as ops
        # with `stack` as a leading literal.
        entity = _find_identity_arg(graph, n)
        if entity is None:
            continue
        # Routing string = the entity's runtime class name, as-is
        # (PascalCase). No identity-class indirection; no lowercase
        # translation. The URL grammar matches the class name.
        object_type_name = type(entity).__name__
        object_id_value = entity.stable_hash
        # Filter precedence: explicit `filter=` literal param wins (mosaic
        # ops take it as a literal arg), else the entity's intrinsic filter
        # (every Stack has exactly one). Centralized in `filter_of`.
        filter_value = filter_of(n, list(n.args) + [entity])
        key = (object_type_name, object_id_value, base_display, filter_value)
        seq[key] = seq.get(key, 0) + 1
        display = (
            base_display if seq[key] == 1
            else f"{base_display} {seq[key]}"
        )
        yield n, (object_type_name, object_id_value, display, filter_value)


# Pin to the head node. trace_pipeline is orchestration, not compute: its
# @node wrapper reads/writes the result to the *local* diskcache of
# whatever node the task lands on, and its result (a Graph) is file-less,
# so it has no shared-disk source of truth to fall back on (unlike
# file-writing ops, which preflight validates against disk). Left to the
# default SPREAD, it can land on a worker and read/write that worker's
# local diskcache — the "silent-write-to-worker-disk" bug ray_init.py
# warns about, which is why the orchestrator actors are head-pinned too.
# Only the head carries the `diskcache` resource, so requesting a sliver
# of it (the same idiom as ContextActor/StateActor) keeps tracing on the
# head against the canonical diskcache.
@node(resources={"diskcache": 0.001})
def trace_pipeline(
    pipeline_name: str,
    pipeline_source_hash: str,
    entity,
    cfg,
):
    """Pure trace of one pipeline. Returns the traced `Graph`, or `None`
    on a trace failure (the pipeline raised, or wasn't registered).

    Cached via @node on (source hash + args). `pipeline_source_hash`
    is in the args so changing the pipeline body invalidates the cache
    — @node alone would only hash THIS function's body, not the
    looked-up pipeline_fn. `cfg` is the FULL ProcessingConfig (frozen
    dataclass, hashable), so dashboard-trace cfg and live-runtime cfg
    naturally get separate cache entries.

    Side-effect-free: tap paths are derived on demand by the dashboard
    (trace + disk), not committed anywhere here.
    """
    from pipelines.registry import registered_pipelines

    pipeline_fn = registered_pipelines().get((pipeline_name, type(entity)))
    if pipeline_fn is None:
        return None

    try:
        return pipeline_fn(entity, cfg)
    except Exception as e:
        import traceback
        print(f"### trace_pipeline failed: {pipeline_name} for {type(entity).__name__}: {e}")
        traceback.print_exc()
        return None


def _decision_closure_ids(entity, entity_id: str, objects_by_type) -> frozenset:
    """The containment closure of a cart entity, as stable-hash ids: the
    entity itself, plus (for a mosaic) its targets and their stacks, plus
    (for a target) its stacks. Pure id-tuple walking — `Mosaic.target_ids`
    and `Target.stack_ids` are plain hash tuples, resolved against the
    build's own `objects` dicts (never live context). Used to scope the
    decisions baked into a pipeline's ProcessingConfig."""
    from cache.context_ref import ContextRef

    def _unwrap(v):
        return v.get() if isinstance(v, ContextRef) else v

    ids = {entity_id}
    sh = getattr(entity, "stable_hash", None)
    if sh:
        ids.add(sh)
    target_ids = tuple(getattr(entity, "target_ids", ()) or ())
    targets_dict = objects_by_type.get("target", {})
    targets = [entity] if hasattr(entity, "stack_ids") else []
    for tid in target_ids:
        ids.add(tid)
        ref = targets_dict.get(tid)
        if ref is not None:
            targets.append(_unwrap(ref))
    for t in targets:
        for sid in tuple(getattr(t, "stack_ids", ()) or ()):
            ids.add(sid)
    return frozenset(ids)


@node
def build_main_processing_graph(
    development_settings: DevelopmentSettings,
    min_frames_in_stack: int,
    decision_overrides,
    shopping_cart: tuple,
    code_version: str,
    objects: Objects,
) -> Graph:
    # `code_version` (the deployed cluster SHA) is a cache-key arg only — the
    # body never reads it. It forces this @node-cached composed graph to
    # re-derive on ANY code change, not just an edit to this builder's own
    # source. Without it, editing an op deep in a pipeline (e.g.
    # `post_process_stack_oss`) leaves the cached graph holding the op's old
    # call_hash, so preflight sees the old outputs as satisfied and nothing
    # rebuilds. Mirrors `trace_pipeline`'s `pipeline_source_hash` arg.
    from decisions.settings import ProcessingConfig
    from frozendict import frozendict as fd

    stacks = objects.stacks
    targets = objects.targets
    mosaics = objects.mosaics
    from cache.context_ref import ContextRef
    cal_stacks_ref = ContextRef.store_large(objects.calibration_stacks)

    log.info("Building main processing graph", shopping_cart=shopping_cart)

    # Cart's URL-friendly type-string → @pipeline-registered class via the
    # canonical `object_class` (the cart and routes use strings; @pipeline
    # decorators key on the actual class).
    from products.product import object_class
    objects_by_type = {"stack": stacks, "target": targets, "mosaic": mosaics}

    # Calibration stacks are SHARED infrastructure: `calibration_plan` runs
    # `resolve_calibration` on cal stacks themselves (a flat resolving its
    # dark), so decisions parented on a cal-stack id are consumed inside any
    # pipeline that might select that master. They join every closure —
    # a cal decision legitimately invalidates all potential consumers.
    cal_stack_ids = frozenset(objects.calibration_stacks.keys())

    def config_for(entry_type, entity, entity_id):
        # Scope the baked decisions to the entity's CONTAINMENT CLOSURE plus
        # the shared cal-stack ids — the cache key of an image may depend
        # only on decisions pertaining to that object, a contained object,
        # or shared calibration. One uniform filter for all entry types
        # (a stack's closure is itself; run-level keys carry the stack as
        # parent, matching the old decisions_for_stack semantics exactly).
        # Baking the GLOBAL dict here (the old mosaic/target behaviour)
        # meant any decision write anywhere re-keyed every mosaic's tail —
        # and the old stack filter had the dual gap: cal-parented decisions
        # were invisible to the ops that consume them.
        closure = (
            frozenset({entity_id})
            if entry_type == "stack"
            else _decision_closure_ids(entity, entity_id, objects_by_type)
        )
        decisions = decisions_for_objects(
            decision_overrides, closure | cal_stack_ids,
        )
        return ProcessingConfig(
            calibration_stacks=cal_stacks_ref,
            development_settings=development_settings,
            min_frames_in_stack=min_frames_in_stack,
            decisions=decisions,
        )

    from pipelines.registry import registered_pipelines
    registered = registered_pipelines()
    graph = Graph({})
    pipeline_tasks = []
    # Eagerly trace each cart entry here on the head-side @node worker.
    # Same call as the lazy run_pipeline does on the cluster; the two
    # traces are bit-identical (deterministic + same context), and
    # structural Node identity means consumer matching against the
    # registry works across the trace/dispatch boundary. The eager
    # pass exists purely to populate the global ephemeral_files
    # registry BEFORE any pipeline's children dispatch — within-cart
    # consumers of a shared ephemeral file see each other in the
    # registry's pending set, so cleanup waits for ALL of them.
    traced: list = []
    for entry in shopping_cart:
        # Cart entries: `(type, id, pipeline_name)`. Each entry
        # becomes a `run_pipeline` Ray task that traces the
        # pipeline body at runtime and returns its Graph as a
        # dynamic sub-graph. The returned Graph carries the
        # tracer's per-pipeline callbacks (`set_workflow_key`),
        # so the per-pipeline workflow-graph state updates fire
        # naturally without any plumbing here.
        type, id, pipeline_name = entry
        if id not in objects_by_type[type]:
            log.warning("Cart entry refers to missing object", entry=entry)
            continue
        if (pipeline_name, object_class(type)) not in registered:
            log.warning("Unknown pipeline in cart", entry=entry)
            continue
        entity = objects_by_type[type][id].get()
        cfg = config_for(type, entity, id)
        task = (run_pipeline, entity, cfg, pipeline_name, type)
        graph[task] = set()
        pipeline_tasks.append(task)
        traced.append(_eager_trace_pipeline(pipeline_name, type, entity, cfg))

    graph[main_processing_graph_done] = pipeline_tasks
    graph.ephemeral_files = _build_ephemeral_registry(traced)
    return graph


def build_compute_graph() -> Graph:
    settings = context.settings
    shopping_cart = tuple(context.shopping_cart or [])

    regroup = (build_regroup_graph, settings.clustering_settings, settings.decision_overrides)

    dev = settings.development_settings
    min_frames = settings.clustering_settings.min_frames_in_stack
    overrides = settings.decision_overrides
    # The deployed cluster SHA — a cache-key arg for the composed graph builders
    # so ANY code change re-derives them (an op edit is otherwise invisible to
    # these @node caches, whose own source is unchanged). "" in dev/tests →
    # identical to the pre-change key.
    from state.code_version import deployed_code_version
    code_version = deployed_code_version()

    graph = {
        regroup: {scatter_compute_image_metadata},
        (build_main_processing_graph, dev, min_frames, overrides, shopping_cart, code_version): {regroup},
    }
    if not dev.skip_preview_graph:
        graph[(build_preview_graph, dev, min_frames, code_version)] = {regroup}

    return Graph(graph)


# TODO: Add a cleanup task that deletes unreferenced stuff from the cache.
# Can I use some kind of automatic ref-counting?
