import ray
from typing import Any
from beartype import beartype

from context import context
from metadata.image_metadata import scatter_compute_image_metadata
from clustering import cluster_by_image_center
from calibration import (
    identify_calibration_stacks,
    associate_calibration_stacks,
)
from previews.stack_thumbnails import scatter_stack_thumbnails
from previews.types import QualityLevel
from mosaics.identification import identify_mosaics
from stacks.compute_graph import build_stacking_subgraph, process_stacks_for_ui
from targets.compute_graph import build_target_subgraph, process_targets_for_ui
from targets.identification import identify_targets
from targets.types import TargetsDict
from mosaics.compute_graph import build_mosaic_subgraph, process_mosaics_for_ui
from mosaics.types import MosaicsDict
from dependency_graph import Graph, Node, chain
from stacks.identification import identify_stacks
from stacks.composite_stack import SimpleStacksDict
from stacks.validation import validate_stacks
from decorators import cached
from customization.compute_graph import (
    scatter_customize_stacks,
    scatter_customize_cal_stacks,
)
from mosaics.compute_graph import stacks_for_mosaic, targets_for_mosaic
from hashing.types import LargeObject
from customization.settings import Settings
from customization.types import CustomizedStacksDict


@ray.remote
def store_calibration_stacks(calibration_stacks: SimpleStacksDict) -> None:
    context.calibration_stacks = calibration_stacks
    print(
        f"### store_calibration_stacks: Stored "
        f"{len(calibration_stacks)} calibration stacks"
    )
    return calibration_stacks


# TODO: Merge combining and storing into a single op.
@ray.remote
def store_stacks(stacks: CustomizedStacksDict):
    assert None not in stacks.values()
    context.stacks = stacks
    print(f"### store_stacks: Stored {len(stacks)} stacks")
    return stacks


@ray.remote
def store_targets(targets: TargetsDict):
    assert None not in targets.values()
    context.targets = targets
    print(f"### store_targets: Stored {len(targets)} targets")
    return targets


@ray.remote
def store_mosaics(mosaics: MosaicsDict):
    assert None not in mosaics.values()
    context.mosaics = mosaics
    print(f"### store_mosaics: Stored {len(mosaics)} mosaics")
    return mosaics


@ray.remote
def combine_dicts(*ancestor_results: tuple[dict[Any, Any]]) -> dict[Any, Any]:
    assert isinstance(ancestor_results, tuple)
    for d in ancestor_results:
        assert None not in d.values(), f"### combine_dicts: None in dict {d}"
    result = {key: value for d in ancestor_results for key, value in d.items()}
    assert None not in result.values()
    return result


@ray.remote
def scatter_spatial_groups(metadata_ref: LargeObject, spatial_groups) -> Graph:
    settings = context.settings or Settings()

    graph = Graph({})

    graph[associate_calibration_stacks] = {
        (
            identify_calibration_stacks,
            metadata_ref,
        )
    }

    quarantine_calibration_stacks_task = (
        scatter_customize_cal_stacks,
        settings,
    )

    graph[quarantine_calibration_stacks_task] = {associate_calibration_stacks}
    graph[store_calibration_stacks] = {quarantine_calibration_stacks_task}

    stack_tasks = set()
    target_tasks = set()
    for i, g in enumerate(spatial_groups):
        stack_task = (
            identify_stacks,
            metadata_ref,
            g,
        )
        validate_stacks_task = Node(validate_stacks, name=str(i))
        customize_stacks_task = Node(
            scatter_customize_stacks, settings, name=str(i)
        )

        target_task = Node(identify_targets, name=str(i))

        # TODO: Validate stacks after customizing them.
        graph[validate_stacks_task] = {stack_task}
        graph[customize_stacks_task] = [
            validate_stacks_task,
            quarantine_calibration_stacks_task,
        ]

        graph[target_task] = {customize_stacks_task}

        stack_tasks.add(customize_stacks_task)
        target_tasks.add(target_task)

    gather_stacks_task = Node(combine_dicts, name="combine_stacks")
    graph[gather_stacks_task] = stack_tasks
    graph[store_stacks] = {gather_stacks_task}
    graph[process_stacks_for_ui] = {gather_stacks_task}

    gather_targets_task = Node(combine_dicts, name="combine_targets")
    graph[gather_targets_task] = target_tasks
    graph[store_targets] = {gather_targets_task}
    graph[process_targets_for_ui] = {gather_targets_task}

    graph[identify_mosaics] = {gather_targets_task}
    graph[store_mosaics] = {identify_mosaics}
    graph[process_mosaics_for_ui] = {identify_mosaics}

    return graph


@ray.remote
def build_regroup_graph(metadata_ref: LargeObject) -> Graph:
    cluster_by_image_center_task = (cluster_by_image_center, metadata_ref)
    scatter_spatial_groups_task = (scatter_spatial_groups, metadata_ref)

    return Graph(
        {
            cluster_by_image_center_task: {},
            scatter_spatial_groups_task: {cluster_by_image_center_task},
        }
    )


@cached.remote
def preview_graph_done() -> None:
    pass


@ray.remote
def build_preview_graph() -> Graph:
    stacks = context.stacks
    targets = context.targets

    tasks = []
    tasks.append((scatter_stack_thumbnails,))

    for mosaic in context.mosaics.values():
        tasks.append(
            (
                build_mosaic_subgraph,
                QualityLevel.SUBFRAME,
                mosaic,
                stacks_for_mosaic(mosaic, stacks, targets),
                targets_for_mosaic(mosaic, targets),
            )
        )

    return Graph({preview_graph_done: tasks})


@ray.remote
def main_processing_graph_done() -> None:
    pass


@ray.remote
def build_main_processing_graph() -> Graph:
    stacks = context.stacks
    targets = context.targets
    mosaics = context.mosaics
    shopping_cart = context.shopping_cart or []

    # TODO: Figure out a nice way to set this during development.
    mode = QualityLevel.MASTER

    print(f"### build_main_processing_graph shopping_cart {shopping_cart}")

    # TODO: Maybe add prioritization options in the UI.
    graph = Graph({})
    tasks = []
    for type, id in shopping_cart:
        if type == "stack":
            if id not in stacks:
                print(
                    f"### WARNING: stack {id} is in shopping cart but does not exist"
                )
                continue
            tasks.append((build_stacking_subgraph, stacks[id]))
        elif type == "target":
            if id not in targets:
                print(
                    f"### WARNING: target {id} is in shopping cart but does not exist"
                )
                continue
            tasks.append((build_target_subgraph, targets[id]))
        if type == "mosaic":
            if id not in mosaics:
                print(
                    f"### WARNING: mosaic {id} is in shopping cart but does not exist"
                )
                continue

            mosaic = mosaics[id]
            tasks.append(
                (
                    build_mosaic_subgraph,
                    mode,
                    mosaic,
                    stacks_for_mosaic(mosaic, stacks, targets),
                    targets_for_mosaic(mosaic, targets),
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
