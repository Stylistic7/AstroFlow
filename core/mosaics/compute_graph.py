import numpy as np
from typing import Optional
from shapely import box  # type: ignore
from shapely.affinity import scale  # type: ignore
from spherical_geometry.polygon import SingleSphericalPolygon
from functools import partial

from context import context
from customization.types import CustomizedStack
from dependency_graph import Graph, Node, chain
from mosaics.merge_images import merge_images
from mosaics.types import Mosaic, MosaicsDict, Viewbox
from mosaics.viewbox import compute_viewbox
from mosaics.placement import place_image
from mosaics.post_process import post_process_mosaic
from stacks.compute_graph import get_stack_image
from stacks.composite_stack import CompositeStacksDict
from stacks.types import StackHash
from previews.types import QualityLevel
from decorators import cached
from targets.types import TargetsDict, Target, TargetHash
from metadata.image_metadata import compute_image_metadata
from stacks.convert_to_dict import convert_to_dict
from mosaics.final_storage import store_mosaic_color_channel
from dependency_graph.workflow_graph_adapter import (
    store_workflow_graph,
    store_node_status,
)


def targets_for_mosaic(mosaic: Mosaic, targets: TargetsDict) -> TargetsDict:
    """Returns only the targets included in the mosaic."""
    return TargetsDict(
        {target_id: targets[target_id] for target_id in mosaic.target_ids}
    )


def stacks_for_mosaic(
    mosaic: Mosaic, stacks: CompositeStacksDict, targets: TargetsDict
) -> CompositeStacksDict:
    """Returns only the stacks included in the mosaic."""
    return CompositeStacksDict(
        {
            stack_id: stacks[stack_id]
            for target_id in mosaic.target_ids
            for stack_id in targets[target_id].stack_ids
        }
    )


@cached.remote
def build_mosaic_subgraph(
    mode: QualityLevel,
    mosaic: Mosaic,
    stacks: CompositeStacksDict,
    targets: TargetsDict,
) -> Graph:
    print(
        f"### build_mosaic_subgraph mode {mode} for mosaic {mosaic} stacks {[str(s) for s in stacks]}"
    )
    viewbox = compute_viewbox(mode, mosaic)
    filters = set(
        stack.filter for stack in stacks.values() if stack is not None
    )

    # TODO: This needs to be moved out of this cached function, into a wrapper.
    if context.override_filters is not None:
        print(f"### WARNING: override_filters: {context.override_filters}")
        filters = context.override_filters

    graph = Graph()

    filter_done_tasks = []
    for filter in filters:
        strips = get_stacks_for_filter(mosaic, filter, targets, stacks)
        print(f"### strips for filter {filter} {strips}")

        if mode == QualityLevel.SUBFRAME:
            # In subframe mode, missing panels are allowed. Just remove them from
            # the strips.
            strips = tuple(
                tuple(s for s in strip if s is not None) for strip in strips
            )
            strips = tuple(s for s in strips if len(s))
        else:
            # In master mode, missing panels are not allowed; skip the filter.
            if any(None in strip for strip in strips):
                print(
                    f"### Skipping mosaic {mosaic} filter {filter} because "
                    f"some stacks are missing"
                )
                continue

        filter_subgraph_task = Node(
            build_mosaic_filter_subgraph,
            mode,
            mosaic,
            strips,
            stacks,
            filter,
            viewbox,
        )
        store_task = (store_mosaic_color_channel, mode, mosaic, filter)
        graph[filter_subgraph_task] = {}
        graph[store_task] = {filter_subgraph_task}
        filter_done_tasks.append(store_task)

    graph[(post_process_mosaic, mode, mosaic)] = filter_done_tasks

    return graph


@cached.remote
def build_placement_subgraph(
    mode: QualityLevel,
    mosaic: Mosaic,
    stack: CustomizedStack,
    filter: str,
    viewbox: Viewbox,
) -> Graph:
    # TODO: Integrate the stack mask here too.
    return Graph(
        {
            **chain(
                (get_stack_image, mode, stack),
                Node(compute_image_metadata, name="stack"),
                (place_image, mode, mosaic, stack, filter, viewbox),
                Node(compute_image_metadata, name="placed"),
            )
        }
    )


def get_stacks_for_filter(
    mosaic: Mosaic,
    filter: str,
    targets: TargetsDict,
    stacks: CompositeStacksDict,
) -> Optional[list[list[StackHash]]]:
    """Returns the mosaics strips with target hashes replaced by stack hashes."""

    def get_stack_for_target(target_id: TargetHash) -> StackHash:
        # TODO: What should be done if there are several stacks in this target
        # with the same filter, but, e.g. different rotations?
        for stack_id in targets[target_id].stack_ids:
            if stacks[stack_id].filter == filter:
                return stack_id

    def get_stacks_for_strip(strip: list[TargetHash]) -> list[StackHash]:
        return tuple(get_stack_for_target(target_id) for target_id in strip)

    return tuple(get_stacks_for_strip(strip) for strip in mosaic.strips)


@cached.remote
def build_mosaic_filter_subgraph(
    mode: QualityLevel,
    mosaic: Mosaic,
    strips: list[list[StackHash]],
    stacks: CompositeStacksDict,
    filter: str,
    viewbox: Viewbox,
) -> Graph:
    print(
        f"### build_mosaic_filter_subgraph mode {mode} "
        f"mosaic {mosaic} filter {filter} strips {mosaic.strips} "
        f"viewbox {viewbox}"
    )

    graph_update_fn, node_status_fn = None, None
    if mode == QualityLevel.MASTER:
        graph_update_fn = partial(
            store_workflow_graph, ("mosaic", mosaic.stable_hash, filter)
        )
        node_status_fn = store_node_status

    # For strip index 0:
    #   merge(place_a, place_b) -> merge_b
    #   merge(merge_b, place_c) -> merge_c
    #   merge(merge_c, place_d) -> merge_d
    #
    # For strip index 1:
    #   merge(place_e, place_f) -> merge_f
    #   merge(merge_f, place_g) -> merge_g
    #   merge(merge_g, place_h) -> merge_h
    #
    # For strip index 2:
    #   merge(place_i, place_j) -> merge_j
    #   merge(merge_j, place_k) -> merge_k
    #   merge(merge_k, place_l) -> merge_l

    graph = Graph(
        graph_update_fn=graph_update_fn, node_status_fn=node_status_fn
    )
    strip_tasks = []
    for strip in strips:
        prev_task = None
        for i, stack_id in enumerate(strip):
            # Missing stacks are acceptable, except in MASTER mode.
            if stack_id is None:
                if mode == QualityLevel.MASTER:
                    print(
                        f"### build_mosaic_filter_subgraph returning None because a "
                        f"stack for filter {filter} was not found"
                    )
                    return None
                else:
                    continue

            stack = stacks[stack_id]
            if i == 0:
                prev_task = Node(
                    build_placement_subgraph,
                    mode,
                    mosaic,
                    stack,
                    filter,
                    viewbox,
                )
            else:
                placement_task = Node(
                    build_placement_subgraph,
                    mode,
                    mosaic,
                    stack,
                    filter,
                    viewbox,
                )
                merge_task = Node(
                    merge_images,
                    mode,
                    mosaic,
                    name=f"merge_{mosaic.stable_hash}_{stack_id}",
                )
                metadata_task = Node(
                    compute_image_metadata,
                    name=f"merge_{mosaic.stable_hash}_{stack_id}",
                )
                if prev_task is not None:
                    graph[placement_task] = [prev_task]
                graph[merge_task] = [prev_task, placement_task]
                graph[metadata_task] = [merge_task]
                prev_task = metadata_task

        assert prev_task is not None, f"strips are {strips}"
        strip_tasks.append(prev_task)

    for i in range(len(strips)):
        if i == 0:
            prev_task = strip_tasks[0]
        else:
            merge_task = Node(
                merge_images,
                mode,
                mosaic,
                name=f"merge_{mosaic.stable_hash}_strip_{i}",
            )
            metadata_task = Node(
                compute_image_metadata,
                name=f"merge_{mosaic.stable_hash}_strip_{i}",
            )
            graph[merge_task] = [prev_task, strip_tasks[i]]
            graph[metadata_task] = [merge_task]
            prev_task = metadata_task

    return graph


@cached.remote
def process_mosaics_for_ui(mosaics: MosaicsDict) -> None:
    def my_transform(v):
        if type(v) is np.ndarray:
            return v.tolist()
        if type(v) is SingleSphericalPolygon:
            return v.points.tolist()
        return v

    def prepare_mosaic(mosaic: Mosaic):
        return convert_to_dict(mosaic, transform=my_transform)

    # TODO: Parallelize this.
    ui_mosaics = {h: prepare_mosaic(s) for h, s in mosaics.items()}
    context.ui_mosaics = ui_mosaics
    return ui_mosaics
