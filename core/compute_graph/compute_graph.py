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
    build_stacking_subgraph,
    process_stacks_for_ui,
    process_calibration_stacks_for_ui,
)
from targets.compute_graph import build_target_subgraph, process_targets_for_ui
from targets.identification import identify_targets
from targets.types import TargetsDict
from mosaics.compute_graph import build_mosaic_subgraph, process_mosaics_for_ui
from mosaics.types import MosaicsDict
from dependency_graph import Graph, Node
from stacks.identification import identify_stacks
from stacks.composite_stack import RunsDict, StacksDict, CalibrationStacksDict
from decorators import node
from decisions.compute_graph import (
    scatter_customize_targets,
)
from mosaics.compute_graph import targets_for_mosaic
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

    tasks = []
    tasks.append((scatter_stack_thumbnails, cal_stacks_ref, objects.stacks))

    for mosaic_ref in objects.mosaics.values():
        mosaic = mosaic_ref.get()
        tasks.append(
            (
                build_mosaic_subgraph,
                QualityLevel.SUBFRAME,
                mosaic,
                targets_for_mosaic(mosaic, objects.targets),
                config,
            )
        )

    return Graph({preview_graph_done: tasks})


# Legacy cart-entry shapes that may still be sitting in
# `context.shopping_cart` after an upgrade. Quality strings → canonical
# pipeline names. Removed once stored state has churned (a session or
# two).
_LEGACY_QUALITY_NAMES = {
    "Subframe", "Master", "DownscaledMaster",
    "PostProcessed", "DownscaledPostProcessed",
}


def _quality_to_pipeline(quality: str, impl: "Implementation") -> str:
    """Back-translate a legacy (quality, impl) pair to a canonical
    pipeline name. PostProcessed / DownscaledPostProcessed don't have
    standalone pipelines yet — fold them into the Master variant."""
    if quality == "Subframe":
        return "Subframe"
    if quality in ("Master", "PostProcessed"):
        return f"{impl.value} Master"
    if quality in ("DownscaledMaster", "DownscaledPostProcessed"):
        return f"{impl.value} Master Downscaled"
    return quality


def _normalize_cart_entry(entry: tuple) -> tuple | None:
    """Map any cart entry shape we've ever shipped to the canonical
    `(type, id, pipeline_name)` 3-tuple, or `None` if it's
    unrecognisable. The shapes seen in the wild:

    * `(type, id, pipeline_name)`            — current (3-tuple)
    * `(type, id, quality_str)`              — legacy (no pipeline)
    * `(type, id, quality_str, impl_str)`    — legacy (4-tuple)
    """
    from previews.types import Implementation
    if len(entry) == 3:
        a, b, c = entry
        if c in _LEGACY_QUALITY_NAMES:
            return (a, b, _quality_to_pipeline(c, Implementation.PIXINSIGHT))
        return entry
    if len(entry) == 4:
        a, b, quality, impl_str = entry
        impl = Implementation(impl_str) if impl_str else Implementation.PIXINSIGHT
        return (a, b, _quality_to_pipeline(quality, impl))
    return None


def _pipeline_to_mode_impl(
    pipeline_name: str,
) -> tuple["QualityLevel", "Implementation"]:
    """Back-translate a pipeline name to the legacy `(quality, impl)`
    pair that the stack/target dispatchers still consume. Mosaic
    dispatch goes via the @pipeline registry directly; this is just
    for stack/target shims that haven't been migrated yet."""
    from previews.types import Implementation
    if pipeline_name == "Subframe":
        return QualityLevel.SUBFRAME, Implementation.PIXINSIGHT
    if pipeline_name == "PixInsight Master":
        return QualityLevel.MASTER, Implementation.PIXINSIGHT
    if pipeline_name == "PixInsight Master Downscaled":
        return QualityLevel.DOWNSCALED_MASTER, Implementation.PIXINSIGHT
    # Fallback for future bundled pipelines ("OSS Master", etc.).
    for impl in Implementation:
        if pipeline_name == f"{impl.value} Master":
            return QualityLevel.MASTER, impl
        if pipeline_name == f"{impl.value} Master Downscaled":
            return QualityLevel.DOWNSCALED_MASTER, impl
    # User-registered pipeline with no quality mapping. Default to
    # MASTER + PIXINSIGHT for the stack/target legacy paths; those
    # paths don't run user pipelines anyway.
    return QualityLevel.MASTER, Implementation.PIXINSIGHT


@ray.remote(num_cpus=0.1)
@beartype
def main_processing_graph_done() -> None:
    pass


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

    from previews.types import Implementation

    def config_for(stack_id=None, mode=QualityLevel.MASTER, implementation=Implementation.PIXINSIGHT):
        if stack_id:
            decisions = decisions_for_stack(decision_overrides, stack_id)
        else:
            decisions = fd(decision_overrides)
        return ProcessingConfig(
            calibration_stacks=cal_stacks_ref,
            development_settings=development_settings,
            min_frames_in_stack=min_frames_in_stack,
            decisions=decisions,
            quality=mode,
            implementation=implementation,
        )

    log.info("Building main processing graph", shopping_cart=shopping_cart)

    graph = Graph({})
    tasks = []
    for raw_entry in shopping_cart:
        # Cart entries: `(type, id, pipeline_name)` 3-tuples. Pipeline
        # name is the unified UI-level concept — bundled built-ins
        # ("PixInsight Master", "PixInsight Master Downscaled",
        # "Subframe", future "OSS Master") and registered user
        # pipelines in `~/AstroFlow/pipelines/` share one namespace.
        # The same mosaic can be added twice with two pipelines for
        # side-by-side comparison; pipeline_name never reaches any
        # cached @node op's args.
        entry = _normalize_cart_entry(raw_entry)
        if entry is None:
            log.warning("Unrecognised cart entry shape", entry=raw_entry)
            continue
        type, id, pipeline_name = entry

        # Stack/target paths haven't been migrated to @pipeline yet, so
        # they consume (quality, impl). Derive from pipeline_name.
        mode, impl = _pipeline_to_mode_impl(pipeline_name)
        # Mosaic path looks up the pipeline_fn from the registry.
        pipeline_fn = None
        if type == "mosaic":
            from pipelines.registry import get_pipeline
            try:
                pipeline_fn = get_pipeline(pipeline_name)
            except KeyError:
                log.warning("Unknown pipeline in cart", pipeline=pipeline_name)
                continue

        if type == "stack":
            if id not in stacks:
                log.warning("Stack in shopping cart does not exist", stack_id=id)
                continue
            tasks.append((build_stacking_subgraph, stacks[id].get(), config_for(id, mode, impl)))
        elif type == "target":
            if id not in targets:
                log.warning("Target in shopping cart does not exist", target_id=id)
                continue
            tasks.append((build_target_subgraph, targets[id].get(), config_for(mode=mode, implementation=impl)))
        elif type == "mosaic":
            if id not in mosaics:
                log.warning("Mosaic in shopping cart does not exist", mosaic_id=id)
                continue
            mosaic = mosaics[id].get()
            cfg = config_for(mode=mode, implementation=impl)
            # Pipelines are graph builders, not data ops — they don't
            # need Ray dispatch. Call directly to get their Graph and
            # splice it into the outer graph (same shape as
            # build_compute_graph is called in-process by state/actor).
            sub_graph = pipeline_fn(mode, mosaic, cfg)
            if sub_graph is None:
                continue
            for node, deps in sub_graph.items():
                graph[node] = deps
            depended_on = {d for ds in sub_graph.values() for d in ds}
            for node in sub_graph:
                if node not in depended_on:
                    tasks.append(node)

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
