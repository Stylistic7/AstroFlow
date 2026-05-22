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
from previews.types import QualityLevel
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
    assert None not in targets.values()
    # targets values are already ContextRefs (from identify_targets)
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
    spatial_groups,
) -> Graph:
    graph = Graph({})

    identify_cal_stacks_task = (
        identify_calibration_stacks,
        metadata_ref,
        clustering_settings.max_run_gap,
        clustering_settings.min_frames_in_stack,
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
    cluster_task = (
        cluster_by_image_center,
        metadata_ref,
        clustering_settings.overlap_fraction,
        clustering_settings.min_component_members,
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
            scatter_task: {cluster_task},
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

    # Auto-preview: the 'Subframe' pipeline is dispatched over every
    # mosaic — semantically "every mosaic is implicitly in the cart
    # with the Subframe pipeline." The user never queues these.
    from pipelines.registry import get_pipeline
    from mosaics.types import Mosaic
    subframe_pipeline = get_pipeline("Subframe", processes=Mosaic)

    graph = Graph({})
    tasks = [(scatter_stack_thumbnails, cal_stacks_ref, objects.stacks)]

    # Subframe pipeline body resolves `mosaic.targets` against
    # `context.targets`; populate it here so the trace runs cleanly.
    from context import local_context as context
    context.targets = objects.targets

    for mosaic_ref in objects.mosaics.values():
        _splice(graph, tasks, subframe_pipeline(mosaic_ref.get(), config))

    graph[preview_graph_done] = tasks
    return graph


def _splice(graph: "Graph", tasks: list, sub_graph) -> None:
    """Splice a pipeline's recorded Graph into the outer graph + add
    its sinks (nodes with no dependents) to the outer task list."""
    if sub_graph is None:
        return
    for node, deps in sub_graph.items():
        graph[node] = deps
    depended_on = {d for ds in sub_graph.values() for d in ds}
    for node in sub_graph:
        if node not in depended_on:
            tasks.append(node)


@ray.remote(num_cpus=0.1)
@beartype
def main_processing_graph_done() -> None:
    pass


def _record_pipeline_taps(record, pipeline_name, cart_type, cart_id, graph):
    """Walk every Node in the pipeline's traced `graph` and record
    tap-point paths in the PipelineOutputIndex.

    A tap point is any @makes_image-decorated op call whose task
    carries a `_display` attribute (set by `@makes_image(display=...)`).
    Intermediate ops (per-strip merge_images, per-panel place_image,
    auto-chained compute_image_metadata) leave no `_display` and are
    skipped.

    For each tap-point Node we:
      1. Bind the Node's literal args to the producer function's
         signature by parameter name.
      2. Try each (identity_cls, level) the @makes_image registered;
         extract that identity's fields from the bound args; if all
         fields are present, call the resolved filename_fn to get
         the on-disk path.
      3. Record (pipeline_name, object_type_from_identity,
         object_id_from_args, display, filter_from_args) → path.

    Filtering by `(cart_type, cart_id)`: callers can be coarse — we
    record EVERY tap whose identity matches; the cart entry's object
    is the canonical id for object-scoped lookups but stack-level
    taps under a mosaic cart entry are also useful (Master per
    constituent stack, etc.), so we don't restrict.
    """
    from dependency_graph.classes import Node
    import dataclasses
    import inspect

    if graph is None:
        return

    for n in list(graph.keys()):
        if not isinstance(n, Node):
            continue
        task = n.task
        # Unwrap Ray remote → underlying function for signature + display.
        fn = getattr(task, "_function", task)
        display = getattr(fn, "_display", None) or getattr(task, "_display", None)
        if display is None:
            continue
        filename_fn = getattr(fn, "filename_fn", None) or getattr(task, "filename_fn", None)
        if filename_fn is None:
            continue
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            continue
        params = list(sig.parameters.keys())
        arg_map = {params[i]: n.args[i]
                   for i in range(min(len(n.args), len(params)))}
        # Try each registered (identity_cls, level) the op declared.
        identities = getattr(fn, "filename_fns", {})
        for (identity_cls, _level) in identities.keys():
            id_field_names = [f.name for f in dataclasses.fields(identity_cls)]
            try:
                id_values = [arg_map[name] for name in id_field_names]
            except KeyError:
                continue
            try:
                path = str(filename_fn(*id_values))
            except (ValueError, TypeError, KeyError):
                continue
            # Extract object_type, object_id, filter for the index key.
            object_type_name = identity_cls.object_type.__name__.lower()
            primary = id_values[0] if id_values else None
            object_id_value = (
                primary.stable_hash if hasattr(primary, "stable_hash")
                else str(primary)
            )
            filter_value = arg_map.get("filter")
            record(pipeline_name, object_type_name, object_id_value,
                   display, filter_value, path)
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

    from pipelines.registry import get_pipeline
    from pipelines.output_index import record as record_pipeline_output
    graph = Graph({})
    tasks = []
    for entry in shopping_cart:
        # Cart entries: `(type, id, pipeline_name)`. Pipelines are
        # graph builders — call in-process, splice the recorded Graph
        # into the outer graph.
        type, id, pipeline_name = entry
        if id not in objects_by_type[type]:
            log.warning("Cart entry refers to missing object", entry=entry)
            continue
        entity = objects_by_type[type][id].get()
        cfg = config_for(id if type == "stack" else None)
        try:
            pipeline_fn = get_pipeline(pipeline_name, processes=type_to_class[type])
        except KeyError:
            log.warning("Unknown pipeline in cart", entry=entry)
            continue
        result = pipeline_fn(entity, cfg)
        _splice(graph, tasks, result)
        # Walk the traced graph for @makes_image(display=...) tap
        # points and index them. View-side endpoints surface every
        # `_display`-tagged Node — this is the only mechanism that
        # scales to user-defined pipelines (no static table edits).
        _record_pipeline_taps(
            record_pipeline_output, pipeline_name, type, id, result,
        )

    graph[main_processing_graph_done] = tasks
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
