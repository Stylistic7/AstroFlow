import ray
import structlog
from typing import Any
from beartype import beartype

from context import context
from cache.context_ref import ContextRef
from metadata.classes import MetadataDict
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
from cache.context_ref import ContextRef
from decisions.settings import ClusteringSettings, DevelopmentSettings
from decisions.decisions import decisions_for_stack
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
    from clustering.fingerprint import FingerprintMap

    # Two ancestor results in non-deterministic order: the spatial-group
    # list from cluster_by_image_center and the global FingerprintMap
    # from identify_camera_fingerprints. Discriminate by type.
    spatial_groups = next(
        r for r in ancestor_results if not isinstance(r, FingerprintMap)
    )
    fingerprint_map = next(
        r for r in ancestor_results if isinstance(r, FingerprintMap)
    )

    graph = Graph({})

    identify_cal_stacks_task = (
        identify_calibration_stacks,
        metadata_ref,
        clustering_settings.max_run_gap,
        clustering_settings.min_frames_in_stack,
        fingerprint_map,
    )

    graph[associate_calibration_stacks] = {identify_cal_stacks_task}
    graph[store_calibration_stacks] = {associate_calibration_stacks}
    graph[process_calibration_stacks_for_ui] = {associate_calibration_stacks}
    graph[store_ui_calibration_stacks] = {process_calibration_stacks_for_ui}

    stack_tasks = set()
    target_tasks = set()
    for i, g in enumerate(spatial_groups):
        stack_task = (
            identify_stacks,
            metadata_ref,
            clustering_settings.angular_tolerance,
            g,
            clustering_settings.min_frames_in_stack,
            clustering_settings.max_run_gap,
            fingerprint_map,
        )

        target_task = Node(identify_targets, clustering_settings.angular_tolerance, name=str(i))
        customize_targets_task = Node(
            scatter_customize_targets, name=str(i)
        )

        # Each target identification depends on its own spatial group's stacks.
        graph[target_task] = {stack_task}
        graph[customize_targets_task] = {target_task}

        stack_tasks.add(stack_task)
        target_tasks.add(customize_targets_task)

    gather_stacks_task = Node(combine_dicts, name="combine_stacks")
    graph[gather_stacks_task] = stack_tasks
    graph[store_stacks] = {gather_stacks_task}
    process_stacks_task = (
        process_stacks_for_ui,
        decision_overrides,
        clustering_settings.min_frames_in_stack,
    )
    graph[process_stacks_task] = {store_stacks, store_calibration_stacks}
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
    from stacks.identification import identify_camera_fingerprints

    cluster_task = (
        cluster_by_image_center,
        metadata_ref,
        clustering_settings.overlap_fraction,
        clustering_settings.min_component_members,
    )
    fingerprint_task = (
        identify_camera_fingerprints,
        metadata_ref,
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
            scatter_task: {cluster_task, fingerprint_task},
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
    objects: Objects,
) -> Graph:
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
    for mosaic_ref in objects.mosaics.values():
        mosaic = mosaic_ref.get()
        task = (run_pipeline, mosaic, config, "Subframe", "mosaic")
        graph[task] = set()
        tasks.append(task)

    graph[preview_graph_done] = tasks
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


@ray.remote(num_cpus=0.1)
def run_pipeline(entity, cfg, pipeline_name: str, type_tag: str) -> "Graph":
    """Trace one pipeline at runtime and return its Graph as a
    dynamic sub-graph.

    Returning a Graph from a `@ray.remote` task is how `run_graph`
    spins up a child `GraphRuntime` for the work the pipeline
    represents. That sub-runtime reads the tracer's per-pipeline
    `node_status_fn` (= `store_node_status`) and `graph_update_fn`
    (= `partial(store_workflow_graph, pipeline_key)`) straight off
    the returned Graph — so per-pipeline live state updates fire
    naturally as nodes complete inside this pipeline's sub-runtime,
    without any extra plumbing at the parent.

    Replaces the old `_splice` pattern, which flattened every
    pipeline's traced Graph into one big parent dict and dropped
    the per-Graph callbacks on the floor."""
    from pipelines.registry import get_pipeline
    from pipelines.output_index import record_existence
    from stacks.composite_stack import Stack
    from targets.types import Target
    from mosaics.types import Mosaic

    type_to_class = {"stack": Stack, "target": Target, "mosaic": Mosaic}
    pipeline_fn = get_pipeline(pipeline_name, processes=type_to_class[type_tag])
    graph = pipeline_fn(entity, cfg)
    if graph is None:
        # Some pipelines (MASTER-mode mosaics with missing panels)
        # legitimately bail. Surface an empty graph so the parent
        # task's `node_completed` still gets a Graph value and
        # closes out the cart entry cleanly.
        from dependency_graph import Graph
        return Graph({})
    # Walk the traced graph for @makes_image(display=...) tap
    # points: register their existence in the PipelineOutputIndex
    # (for dashboard discovery) AND splice in the runtime
    # `_register_tap_path_op` Nodes that record each tap's actual
    # on-disk path once the producing op completes.
    _record_pipeline_taps(
        record_existence, pipeline_name, type_tag, entity.stable_hash, graph,
    )
    # GraphRuntime asserts a sub-graph has exactly one terminal
    # node. Funnel every current terminal into a sentinel so the
    # child sub-runtime has a single completion handle (parent's
    # `run_pipeline` task is `mark_done`'d with this sentinel's
    # result — None, since the parent only cares about ordering).
    depended_on = {d for ds in graph.values() for d in ds}
    terminals = [n for n in graph if n not in depended_on]
    graph[run_pipeline_done] = terminals
    return graph


@ray.remote(num_cpus=0.1)
@beartype
def main_processing_graph_done() -> None:
    pass


@ray.remote(num_cpus=0.01)
def _register_tap_path_op(
    pipeline_name: str, object_type: str, object_id: str,
    display: str, filter: str | None, result,
):
    """Runtime helper: record the tap's actual on-disk path in the
    PipelineOutputIndex after the upstream @makes_image op has
    finished writing.

    `result` is the resolved upstream Node value — a `WithSidecar`
    or `ImageMetadata` carrying a `.filename` attribute. The path
    flows in via Ray's data-dep mechanism, so it's whatever the op
    body actually wrote to (no trace-time prediction needed)."""
    from pipelines.output_index import record_path
    path = getattr(result, "filename", None) or result
    record_path(
        pipeline_name, object_type, object_id, display, filter, str(path),
    )
    return result


def _record_pipeline_taps(record_existence, pipeline_name, cart_type, cart_id, graph):
    """Walk every Node in the pipeline's traced `graph`. For each
    @makes_image tap point (Nodes whose task carries `_display`):

    1. Register the tap's EXISTENCE in the PipelineOutputIndex
       (`(pipeline, type, id, display, filter) → "pending"`). The UI
       can render placeholders immediately.

    2. Add a downstream `_register_tap_path_op` Node that depends on
       the tap's result. When the tap finishes and the path is
       known, this follow-up writes the real path to the index.

    Same-key collisions (e.g. a pipeline that calls an atomic op
    like `merge_images` more than once on the same (mosaic, filter))
    are disambiguated by suffixing ` 2`, ` 3`, … to the display
    string in the order the Nodes appear in the trace. The first
    occurrence keeps the unmodified display name — single-merge
    pipelines aren't cluttered with "Merged 1" when there's no
    ambiguity to resolve.

    The two-phase pattern (existence at trace, path at runtime) is
    the framework's answer to "shared ops across pipelines must
    produce distinct files even when literals match" — each tap's
    on-disk filename is chosen by the op body using its full bound
    args (including resolved data deps), and the runtime hook
    captures that filename regardless of how the body computed it.
    """
    from dependency_graph.classes import Node
    import dataclasses
    import inspect

    if graph is None:
        return

    # Per-(type, id, display, filter) sequence count, so duplicate
    # taps in the same trace get distinct display names.
    seq: dict[tuple, int] = {}

    for n in list(graph.keys()):
        if not isinstance(n, Node):
            continue
        task = n.task
        fn = getattr(task, "_function", task)
        base_display = getattr(fn, "_display", None) or getattr(task, "_display", None)
        if base_display is None:
            continue
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            continue
        params = list(sig.parameters.keys())
        arg_map = {params[i]: n.args[i]
                   for i in range(min(len(n.args), len(params)))}
        identities = getattr(fn, "_identities", None) or getattr(task, "_identities", [])
        for identity_cls in identities:
            id_field_names = [f.name for f in dataclasses.fields(identity_cls)]
            try:
                id_values = [arg_map[name] for name in id_field_names]
            except KeyError:
                continue
            object_type_name = identity_cls.object_type.__name__.lower()
            primary = id_values[0] if id_values else None
            object_id_value = (
                primary.stable_hash if hasattr(primary, "stable_hash")
                else str(primary)
            )
            filter_value = arg_map.get("filter")
            key = (object_type_name, object_id_value, base_display, filter_value)
            seq[key] = seq.get(key, 0) + 1
            display = (
                base_display if seq[key] == 1
                else f"{base_display} {seq[key]}"
            )
            record_existence(
                pipeline_name, object_type_name, object_id_value,
                display, filter_value,
            )
            # Schedule a runtime task that, when n's result is
            # available, writes the real path to the tap index. The
            # task depends on n via the graph adjacency (one entry,
            # ordering = data flow). Its literal args bake in the
            # identity; its data-dep is the tap's result.
            register_node = Node(
                _register_tap_path_op,
                pipeline_name, object_type_name, object_id_value,
                display, filter_value,
                name=f"_register_tap_{n.name}",
            )
            graph[register_node] = [n]
            break  # one identity matched; don't double-record


@node
def build_main_processing_graph(
    development_settings: DevelopmentSettings,
    min_frames_in_stack: int,
    decision_overrides,
    shopping_cart: tuple,
    objects: Objects,
) -> Graph:
    from decisions.settings import ProcessingConfig
    from frozendict import frozendict as fd

    stacks = objects.stacks
    targets = objects.targets
    mosaics = objects.mosaics
    from cache.context_ref import ContextRef
    cal_stacks_ref = ContextRef.store_large(objects.calibration_stacks)

    log.info("Building main processing graph", shopping_cart=shopping_cart)

    # Cart's URL-friendly type-string → @pipeline-registered class.
    # The cart and routes use strings; @pipeline decorators key on
    # the actual class (typo-safe, self-documenting in pipeline
    # bodies). This dict is the only place those conventions meet.
    from stacks.composite_stack import Stack
    from targets.types import Target
    from mosaics.types import Mosaic
    type_to_class = {"stack": Stack, "target": Target, "mosaic": Mosaic}
    objects_by_type = {"stack": stacks, "target": targets, "mosaic": mosaics}

    def config_for(stack_id=None):
        decisions = (
            decisions_for_stack(decision_overrides, stack_id) if stack_id
            else fd(decision_overrides)
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
        if (pipeline_name, type_to_class[type]) not in registered:
            log.warning("Unknown pipeline in cart", entry=entry)
            continue
        entity = objects_by_type[type][id].get()
        cfg = config_for(id if type == "stack" else None)
        task = (run_pipeline, entity, cfg, pipeline_name, type)
        graph[task] = set()
        pipeline_tasks.append(task)

    graph[main_processing_graph_done] = pipeline_tasks
    return graph


def build_compute_graph() -> Graph:
    settings = context.settings
    shopping_cart = tuple(context.shopping_cart or [])

    regroup = (build_regroup_graph, settings.clustering_settings, settings.decision_overrides)

    dev = settings.development_settings
    min_frames = settings.clustering_settings.min_frames_in_stack
    overrides = settings.decision_overrides

    graph = {
        regroup: {scatter_compute_image_metadata},
        (build_main_processing_graph, dev, min_frames, overrides, shopping_cart): {regroup},
    }
    if not dev.skip_preview_graph:
        graph[(build_preview_graph, dev, min_frames)] = {regroup}

    return Graph(graph)


# TODO: Add a cleanup task that deletes unreferenced stuff from the cache.
# Can I use some kind of automatic ref-counting?
