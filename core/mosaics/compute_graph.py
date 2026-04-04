import ray
import numpy as np
import structlog
from typing import Optional
from shapely import box  # type: ignore
from shapely.affinity import scale  # type: ignore
from spherical_geometry.polygon import SingleSphericalPolygon
from functools import partial

from frozendict import frozendict
from stacks.composite_stack import Stack, StacksDict, RunsDict, CalibrationStacksDict
from decisions.settings import DevelopmentSettings
from dependency_graph import Graph, Node, chain
from mosaics.merge_images import merge_images
from mosaics.types import Mosaic, MosaicsDict, Viewbox
from mosaics.viewbox import compute_viewbox
from mosaics.placement import place_image
from mosaics.post_process import post_process_mosaic
from stacks.compute_graph import get_stack_image
from stacks.types import StackHash
from previews.types import QualityLevel
from decorators import node
from targets.types import TargetsDict, Target, TargetHash
from metadata.image_metadata import compute_image_metadata
from stacks.convert_to_dict import convert_to_dict
from mosaics.final_storage import store_mosaic_color_channel
from dependency_graph.workflow_graph_adapter import (
    store_workflow_graph,
    store_node_status,
)

log = structlog.get_logger()


def targets_for_mosaic(mosaic: Mosaic, targets: TargetsDict) -> TargetsDict:
    """Returns only the targets included in the mosaic."""
    return TargetsDict(
        {target_id: targets[target_id] for target_id in mosaic.target_ids}
    )


def _unwrap(ref):
    """Unwrap a ContextRef to its value, or return plain objects as-is."""
    from cache.context_ref import ContextRef
    return ref.get() if isinstance(ref, ContextRef) else ref


def stacks_for_mosaic(
    mosaic: Mosaic, stacks: StacksDict, targets: TargetsDict
) -> StacksDict:
    """Returns only the stacks included in the mosaic."""
    return StacksDict(
        {
            stack_id: stacks[stack_id]
            for target_id in mosaic.target_ids
            for stack_id in _unwrap(targets[target_id]).stack_ids
        }
    )


@ray.remote(num_cpus=0.1)
def build_mosaic_subgraph(
    mode: QualityLevel,
    mosaic: Mosaic,
    stacks: StacksDict,
    targets: TargetsDict,
    config: "ProcessingConfig",
) -> Graph:
    log.info("Building mosaic subgraph", mode=mode, mosaic=str(mosaic), stacks=[str(s) for s in stacks])
    viewbox = compute_viewbox(mode, mosaic, config.downscale_factor)
    filters = set(
        ref.get().filter for ref in stacks.values() if ref is not None
    )

    if config.development_settings.override_filters is not None:
        log.warning("Overriding filters", override_filters=config.development_settings.override_filters)
        filters = config.development_settings.override_filters

    graph = Graph()

    filter_done_tasks = []
    for filter in filters:
        strips = get_stacks_for_filter(mosaic, filter, targets, stacks)
        log.debug("Strips for filter", filter=filter, strips=strips)

        if mode == QualityLevel.SUBFRAME:
            # In subframe mode, missing panels are allowed. Just remove them from
            # the strips.
            strips = tuple(
                tuple(s for s in strip if s is not None) for strip in strips
            )
            strips = tuple(s for s in strips if len(s))
            if not strips:
                log.warning("Skipping mosaic filter, all stacks missing", mosaic=str(mosaic), filter=filter)
                continue
        else:
            # In master mode, missing panels are not allowed; skip the filter.
            if any(None in strip for strip in strips):
                log.warning("Skipping mosaic filter, some stacks missing", mosaic=str(mosaic), filter=filter)
                continue

        filter_subgraph_task = Node(
            build_mosaic_filter_subgraph,
            mode,
            mosaic,
            strips,
            stacks,
            filter,
            viewbox,
            config,
        )
        store_task = (store_mosaic_color_channel, mode, mosaic, filter)
        graph[filter_subgraph_task] = {}
        graph[store_task] = {filter_subgraph_task}
        filter_done_tasks.append(store_task)

    graph[(post_process_mosaic, mode, mosaic, stacks, targets)] = filter_done_tasks

    return graph


@ray.remote(num_cpus=0.1)
def build_placement_subgraph(
    mode: QualityLevel,
    mosaic: Mosaic,
    stack: Stack,
    filter: str,
    viewbox: Viewbox,
    config: "ProcessingConfig",
) -> Graph:
    return Graph(
        {
            **chain(
                (get_stack_image, mode, stack, config),
                (place_image, mode, mosaic, stack, filter, viewbox),
                Node(compute_image_metadata, config.development_settings, name="placed"),
            )
        }
    )


def get_stacks_for_filter(
    mosaic: Mosaic,
    filter: str,
    targets: TargetsDict,
    stacks: StacksDict,
) -> Optional[list[list[StackHash]]]:
    """Returns the mosaics strips with target hashes replaced by stack hashes."""

    def get_stack_for_target(target_id: TargetHash) -> StackHash:
        # TODO: What should be done if there are several stacks in this target
        # with the same filter, but, e.g. different rotations?
        for stack_id in _unwrap(targets[target_id]).stack_ids:
            if _unwrap(stacks[stack_id]).filter == filter:
                return stack_id

    def get_stacks_for_strip(strip: list[TargetHash]) -> list[StackHash]:
        return tuple(get_stack_for_target(target_id) for target_id in strip)

    return tuple(get_stacks_for_strip(strip) for strip in mosaic.strips)


@ray.remote(num_cpus=0.1)
def build_mosaic_filter_subgraph(
    mode: QualityLevel,
    mosaic: Mosaic,
    strips: list[list[StackHash]] | tuple,
    stacks: StacksDict | dict | object,
    filter: str,
    viewbox: Viewbox | None,
    config: "ProcessingConfig",
) -> Graph | None:
    log.info("Building mosaic filter subgraph", mode=mode, mosaic=str(mosaic), filter=filter, strips=mosaic.strips, viewbox=str(viewbox))

    graph_update_fn = partial(
        store_workflow_graph, ("mosaic", mosaic.stable_hash, filter, mode.value)
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
                    log.warning("Returning None, stack for filter not found", filter=filter)
                    return None
                else:
                    continue

            stack = _unwrap(stacks[stack_id])
            if i == 0:
                prev_task = Node(
                    build_placement_subgraph,
                    mode, mosaic, stack, filter, viewbox, config,
                )
            else:
                placement_task = Node(
                    build_placement_subgraph,
                    mode, mosaic, stack, filter, viewbox, config,
                )
                merge_task = Node(
                    merge_images,
                    mode,
                    mosaic,
                    name=f"merge_{mosaic.stable_hash}_{stack_id}",
                )
                metadata_task = Node(
                    compute_image_metadata, config.development_settings,
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
                compute_image_metadata, config.development_settings,
                name=f"merge_{mosaic.stable_hash}_strip_{i}",
            )
            graph[merge_task] = [prev_task, strip_tasks[i]]
            graph[metadata_task] = [merge_task]
            prev_task = metadata_task

    # If prev_task was never added to the graph as a key (single-strip, single-stack
    # case), register it as a leaf so the subgraph actually runs it.
    if strip_tasks and prev_task not in graph:
        graph[prev_task] = []

    return graph


@node
def process_mosaics_for_ui(mosaics: MosaicsDict) -> dict:
    def my_transform(v):
        if type(v) is np.ndarray:
            return v.tolist()
        if type(v) is SingleSphericalPolygon:
            return v.points.tolist()
        return v

    def prepare_mosaic(mosaic: Mosaic):
        return convert_to_dict(mosaic, transform=my_transform)

    # TODO: Parallelize this.
    ui_mosaics = {h: prepare_mosaic(ref.get()) for h, ref in mosaics.items()}
    return ui_mosaics
