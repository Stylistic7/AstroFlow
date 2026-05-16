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
from mosaics.merge_images import MergedImage
from mosaics.types import Mosaic, MosaicsDict, Viewbox
from mosaics.viewbox import compute_viewbox
from mosaics.placement import PlacedImage
from stacks.types import StackHash
from previews.types import QualityLevel
from products import get_producer, MediaType
from decorators import node
from targets.types import TargetsDict, Target, TargetHash
from targets.target_channel import (
    TargetChannel, build_target_channel_subgraph,
    channel_stacks_for, filters_for_target,
)
from metadata.image_metadata import compute_image_metadata
from stacks.convert_to_dict import convert_to_dict
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


@ray.remote(num_cpus=0.1)
def build_mosaic_subgraph(
    mode: QualityLevel,
    mosaic: Mosaic,
    targets: TargetsDict,
    config: "ProcessingConfig",
) -> Graph:
    log.info("Building mosaic subgraph", mode=mode, mosaic=str(mosaic))
    viewbox = compute_viewbox(mode, mosaic)

    # Normalize: callers (build_preview_graph, _process_mosaic) construct
    # config without setting `quality`, leaving it at the dataclass default
    # (MASTER). When that config flows down to build_target_channel_subgraph,
    # the SUBFRAME short-circuit doesn't fire and we kick off MASTER-level
    # stacking for every (target, filter) in the mosaic — even on a
    # SUBFRAME preview pass with an empty shopping cart. Make `mode` the
    # source of truth here so downstream code can rely on `config.quality`.
    from dataclasses import replace
    config = replace(config, quality=mode)

    filters: set[str] = set()
    for tid in mosaic.target_ids:
        filters.update(filters_for_target(_unwrap(targets[tid])))

    if config.development_settings.override_filters is not None:
        log.warning("Overriding filters", override_filters=config.development_settings.override_filters)
        filters = config.development_settings.override_filters

    graph = Graph()

    filter_done_tasks = []
    for filter in filters:
        strips = get_targets_for_filter(mosaic, filter, targets)

        if mode == QualityLevel.SUBFRAME:
            # In subframe mode, missing panels are allowed. Just remove them from
            # the strips.
            strips = tuple(
                tuple(t for t in strip if t is not None) for strip in strips
            )
            strips = tuple(s for s in strips if len(s))
            if not strips:
                log.warning("Skipping mosaic filter, all targets missing", mosaic=str(mosaic), filter=filter)
                continue
        else:
            # Master mode: the mosaic merge needs every panel, so a
            # missing one means we can't produce the mosaic in this
            # filter. Still kick off per-(target, filter) merges for
            # the panels we DO have, so their masters are ready for
            # the next time the user images.
            if any(None in strip for strip in strips):
                log.warning(
                    "Mosaic incomplete in this filter; merging present panels only",
                    mosaic=str(mosaic), filter=filter,
                )
                for strip in strips:
                    for target in strip:
                        if target is None:
                            continue
                        filter_done_tasks.append(
                            (build_target_channel_subgraph, target, filter,
                             channel_stacks_for(target, filter), config)
                        )
                continue

        filter_subgraph_task = Node(
            build_mosaic_filter_subgraph,
            mode,
            mosaic,
            strips,
            filter,
            viewbox,
            config,
        )
        from mosaics.final_storage import store_mosaic_color_channel
        store_task = (store_mosaic_color_channel, mode, mosaic, filter, config.implementation)
        graph[filter_subgraph_task] = {}
        graph[store_task] = {filter_subgraph_task}
        filter_done_tasks.append(store_task)

    pp_fn = get_producer(MediaType.IMAGE, Mosaic, QualityLevel.POST_PROCESSED, config.implementation)
    graph[(pp_fn, mode, mosaic, targets)] = filter_done_tasks

    return graph


@ray.remote(num_cpus=0.1)
def build_placement_subgraph(
    mode: QualityLevel,
    mosaic: Mosaic,
    target: Target,
    filter: str,
    viewbox: Viewbox,
    config: "ProcessingConfig",
) -> Graph:
    from products import ensure_producers_registered
    ensure_producers_registered()

    place_fn = get_producer(MediaType.IMAGE, PlacedImage, mode, config.implementation)
    channel_stacks = channel_stacks_for(target, filter)
    # build_target_channel_subgraph already terminates in
    # compute_image_metadata, so place_fn consumes that ImageMetadata
    # directly. The trailing compute_image_metadata is for the placed
    # tiff that place_fn produces. Name must disambiguate this
    # placement from every other (target, filter) placement —
    # compute_image_metadata takes no positional args, so two Nodes
    # with the same name collapse into one in the outer graph and
    # all placements end up feeding into a single metadata node.
    return Graph(
        {
            **chain(
                (build_target_channel_subgraph, target, filter, channel_stacks, config),
                (place_fn, mode, mosaic, target, filter, viewbox),
                Node(
                    compute_image_metadata,
                    name=f"placed_{mosaic.stable_hash}_{target.stable_hash}_{filter}",
                ),
            )
        }
    )


def get_targets_for_filter(
    mosaic: Mosaic,
    filter: str,
    targets: TargetsDict,
) -> tuple[tuple[Target | None, ...], ...]:
    """Replace each TargetHash in mosaic.strips with the resolved Target,
    or None if the target has no stack matching this filter.
    """
    def target_with_filter(target_id: TargetHash) -> Target | None:
        target = _unwrap(targets[target_id])
        return target if filter in filters_for_target(target) else None

    return tuple(
        tuple(target_with_filter(tid) for tid in strip)
        for strip in mosaic.strips
    )


@ray.remote(num_cpus=0.1)
def build_mosaic_filter_subgraph(
    mode: QualityLevel,
    mosaic: Mosaic,
    strips: tuple[tuple[Target | None, ...], ...] | tuple,
    filter: str,
    viewbox: Viewbox | None,
    config: "ProcessingConfig",
) -> Graph | None:
    from products import ensure_producers_registered
    ensure_producers_registered()

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

    merge_fn = get_producer(MediaType.IMAGE, MergedImage, mode, config.implementation)

    graph = Graph(
        graph_update_fn=graph_update_fn, node_status_fn=node_status_fn
    )
    strip_tasks = []
    for strip in strips:
        prev_task = None
        for i, target in enumerate(strip):
            # Missing targets are acceptable, except in MASTER mode.
            if target is None:
                if mode == QualityLevel.MASTER:
                    log.warning("Returning None, target for filter not found", filter=filter)
                    return None
                else:
                    continue

            if i == 0:
                prev_task = Node(
                    build_placement_subgraph,
                    mode, mosaic, target, filter, viewbox, config,
                )
            else:
                placement_task = Node(
                    build_placement_subgraph,
                    mode, mosaic, target, filter, viewbox, config,
                )
                merge_task = Node(
                    merge_fn,
                    mode,
                    mosaic,
                    filter,
                    config.implementation,
                    config.development_settings.mock_pixinsight,
                    name=f"merge_{mosaic.stable_hash}_{filter}_{target.stable_hash}",
                )
                metadata_task = Node(
                    compute_image_metadata,
                    name=f"merge_{mosaic.stable_hash}_{filter}_{target.stable_hash}",
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
                merge_fn,
                mode,
                mosaic,
                filter,
                config.implementation,
                config.development_settings.mock_pixinsight,
                name=f"merge_{mosaic.stable_hash}_{filter}_strip_{i}",
            )
            metadata_task = Node(
                compute_image_metadata,
                name=f"merge_{mosaic.stable_hash}_{filter}_strip_{i}",
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
    ui_mosaics = {h: prepare_mosaic(_unwrap(ref)) for h, ref in mosaics.items()}
    return ui_mosaics
