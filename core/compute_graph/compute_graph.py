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
from dependency_graph import Graph, Node, chain
from stacks.identification import identify_stacks
from stacks.composite_stack import SimpleStacksDict, CompositeStacksDict
from stacks.validation import validate_stacks
from decorators import node
from customization.compute_graph import (
    scatter_customize_stacks,
    scatter_customize_cal_stacks,
    scatter_customize_targets,
)
from mosaics.compute_graph import stacks_for_mosaic, targets_for_mosaic
from cache.context_ref import ContextRef
from customization.settings import Settings

log = structlog.get_logger()


@ray.remote
@beartype
def store_calibration_stacks(
    calibration_stacks: SimpleStacksDict,
) -> SimpleStacksDict:
    context.calibration_stacks = calibration_stacks
    log.info("Stored calibration stacks", count=len(calibration_stacks))
    return calibration_stacks


# TODO: Merge combining and storing into a single op.
@ray.remote
@beartype
def store_stacks(stacks: dict[str, Any]) -> dict[str, Any]:
    assert None not in stacks.values()
    # stacks values are already ContextRefs (from gather_enriched_stacks)
    result = CompositeStacksDict(stacks)
    context.stacks = result
    log.info("Stored stacks", count=len(stacks))
    return result


@ray.remote
@beartype
def store_targets(targets: dict[str, Any]) -> dict[str, Any]:
    assert None not in targets.values()
    # targets values are already ContextRefs (from identify_targets)
    result = TargetsDict(targets)
    context.targets = result
    log.info("Stored targets", count=len(targets))
    return result


@ray.remote
@beartype
def store_mosaics(mosaics: MosaicsDict) -> MosaicsDict:
    assert None not in mosaics.values()
    # mosaics values are already ContextRefs (from identify_mosaics)
    context.mosaics = mosaics
    log.info("Stored mosaics", count=len(mosaics))
    return mosaics


@ray.remote
def store_ui_stacks(ui_stacks: dict) -> None:
    context.ui_stacks = ui_stacks


@ray.remote
def store_ui_calibration_stacks(ui_cal_stacks: dict) -> None:
    context.ui_calibration_stacks = ui_cal_stacks


@ray.remote
def store_ui_targets(ui_targets: dict) -> None:
    context.ui_targets = ui_targets


@ray.remote
def store_ui_mosaics(ui_mosaics: dict) -> None:
    context.ui_mosaics = ui_mosaics


@ray.remote
@beartype
def combine_dicts(*ancestor_results: dict[Any, Any]) -> dict[Any, Any]:
    assert isinstance(ancestor_results, tuple)
    for d in ancestor_results:
        assert None not in d.values(), f"### combine_dicts: None in dict {d}"
    result = {key: value for d in ancestor_results for key, value in d.items()}
    assert None not in result.values()
    return result


@ray.remote
def scatter_spatial_groups(
    metadata_ref: ContextRef[MetadataDict], settings: Settings, spatial_groups
) -> Graph:
    cs = settings.clustering_settings

    context.stacks = CompositeStacksDict()
    context.calibration_stacks = SimpleStacksDict()
    context.targets = TargetsDict()
    context.mosaics = MosaicsDict({})  # empty dict of refs

    graph = Graph({})

    graph[associate_calibration_stacks] = {
        (
            identify_calibration_stacks,
            metadata_ref,
            cs.max_run_gap,
        )
    }

    quarantine_calibration_stacks_task = (scatter_customize_cal_stacks,)

    graph[quarantine_calibration_stacks_task] = {associate_calibration_stacks}
    graph[process_calibration_stacks_for_ui] = {
        quarantine_calibration_stacks_task
    }
    graph[store_ui_calibration_stacks] = {process_calibration_stacks_for_ui}
    graph[store_calibration_stacks] = {quarantine_calibration_stacks_task}

    stack_tasks = set()
    target_tasks = set()
    for i, g in enumerate(spatial_groups):
        stack_task = (
            identify_stacks,
            metadata_ref,
            cs.angular_tolerance,
            g,
            cs.min_frames_in_stack,
            cs.max_run_gap,
        )
        validate_stacks_task = Node(validate_stacks, name=str(i))
        customize_stacks_task = Node(
            scatter_customize_stacks,
            settings.stack_settings,
            settings.decision_overrides,
            cs.min_frames_in_stack,
            name=str(i),
        )

        target_task = Node(identify_targets, cs.angular_tolerance, name=str(i))
        customize_targets_task = Node(
            scatter_customize_targets, settings.target_settings, name=str(i)
        )

        # TODO: Validate stacks after customizing them.
        graph[validate_stacks_task] = {stack_task}
        graph[customize_stacks_task] = [
            validate_stacks_task,
            quarantine_calibration_stacks_task,
        ]

        graph[target_task] = {customize_stacks_task}
        graph[customize_targets_task] = {target_task}

        stack_tasks.add(customize_stacks_task)
        target_tasks.add(customize_targets_task)

    gather_stacks_task = Node(combine_dicts, name="combine_stacks")
    graph[gather_stacks_task] = stack_tasks
    graph[store_stacks] = {gather_stacks_task}
    graph[process_stacks_for_ui] = {gather_stacks_task}
    graph[store_ui_stacks] = {process_stacks_for_ui}

    gather_targets_task = Node(combine_dicts, name="combine_targets")
    graph[gather_targets_task] = target_tasks
    graph[store_targets] = {gather_targets_task}
    graph[process_targets_for_ui] = {gather_targets_task}
    graph[store_ui_targets] = {process_targets_for_ui}

    mosaics_task = Node(identify_mosaics, cs.angular_tolerance)
    graph[mosaics_task] = {gather_targets_task}
    graph[store_mosaics] = {mosaics_task}
    graph[process_mosaics_for_ui] = {mosaics_task}
    graph[store_ui_mosaics] = {process_mosaics_for_ui}

    return graph


@ray.remote
@beartype
def build_regroup_graph(metadata_ref: ContextRef[MetadataDict]) -> Graph:
    settings = context.settings
    cs = settings.clustering_settings
    cluster_by_image_center_task = (
        cluster_by_image_center,
        metadata_ref,
        cs.cosine_similarity_threshold,
        cs.min_component_members,
    )
    scatter_spatial_groups_task = (
        scatter_spatial_groups,
        metadata_ref,
        settings,
    )

    return Graph(
        {
            cluster_by_image_center_task: {},
            scatter_spatial_groups_task: {cluster_by_image_center_task},
        }
    )


@node
def preview_graph_done() -> None:
    pass


@ray.remote
@beartype
def build_preview_graph() -> Graph:
    stacks = context.stacks
    targets = context.targets
    calibration_stacks = context.calibration_stacks
    settings = context.settings
    development_settings = settings.development_settings
    min_frames_in_stack = settings.clustering_settings.min_frames_in_stack

    tasks = []
    tasks.append((scatter_stack_thumbnails,))

    for mosaic_ref in context.mosaics.values():
        mosaic = mosaic_ref.get()
        tasks.append(
            (
                build_mosaic_subgraph,
                QualityLevel.SUBFRAME,
                mosaic,
                stacks_for_mosaic(mosaic, stacks, targets),
                targets_for_mosaic(mosaic, targets),
                calibration_stacks,
                development_settings,
                min_frames_in_stack,
                settings.decision_overrides,
            )
        )

    return Graph({preview_graph_done: tasks})


@ray.remote
@beartype
def main_processing_graph_done() -> None:
    pass


@ray.remote
@beartype
def build_main_processing_graph() -> Graph:
    stacks = context.stacks
    targets = context.targets
    mosaics = context.mosaics
    shopping_cart = context.shopping_cart or []
    calibration_stacks = context.calibration_stacks
    settings = context.settings
    development_settings = settings.development_settings
    min_frames_in_stack = settings.clustering_settings.min_frames_in_stack

    # TODO: Figure out a nice way to set this during development.
    mode = QualityLevel.MASTER

    log.info("Building main processing graph", shopping_cart=shopping_cart)

    # TODO: Maybe add prioritization options in the UI.
    graph = Graph({})
    tasks = []
    for type, id in shopping_cart:
        if type == "stack":
            if id not in stacks:
                log.warning("Stack in shopping cart does not exist", stack_id=id)
                continue
            from frozendict import frozendict as fd
            stack_decisions = fd({k: v for k, v in settings.decision_overrides.items() if len(k) >= 3 and k[2] == id})
            tasks.append((build_stacking_subgraph, stacks[id].get(), calibration_stacks, development_settings, min_frames_in_stack, stack_decisions))
        elif type == "target":
            if id not in targets:
                log.warning("Target in shopping cart does not exist", target_id=id)
                continue
            tasks.append((build_target_subgraph, targets[id].get(), development_settings, min_frames_in_stack))
        if type == "mosaic":
            if id not in mosaics:
                log.warning("Mosaic in shopping cart does not exist", mosaic_id=id)
                continue

            mosaic = mosaics[id].get()
            tasks.append(
                (
                    build_mosaic_subgraph,
                    mode,
                    mosaic,
                    stacks_for_mosaic(mosaic, stacks, targets),
                    targets_for_mosaic(mosaic, targets),
                    calibration_stacks,
                    development_settings,
                    min_frames_in_stack,
                    settings.decision_overrides,
                )
            )

    graph[main_processing_graph_done] = tasks

    return graph


def build_compute_graph() -> Graph:
    return Graph(
        {
            **chain(
                scatter_compute_image_metadata,
                build_regroup_graph,
                build_preview_graph,
                build_main_processing_graph,
            )
        }
    )


# TODO: Add a cleanup task that deletes unreferenced stuff from the cache.
# Can I use some kind of automatic ref-counting?
