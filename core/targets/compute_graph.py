import ray
import structlog
import numpy as np
from beartype import beartype
from spherical_geometry.polygon import SingleSphericalPolygon
from functools import partial

from context import context
from decorators import node
from dependency_graph import Graph
from targets.types import Target
from products import get_producer, MediaType
from previews.types import QualityLevel
from targets.types import TargetsDict
from targets.target_channel import (
    TargetChannel, build_target_channel_subgraph, group_stacks_by_filter,
)
from stacks.convert_to_dict import convert_to_dict
from decisions.settings import DevelopmentSettings, ProcessingConfig
from dependency_graph.workflow_graph_adapter import (
    store_workflow_graph,
    store_node_status,
)

log = structlog.get_logger()


@ray.remote(num_cpus=0.1)
@beartype
def build_target_subgraph(
    target: Target,
    config: ProcessingConfig,
) -> Graph:
    log.info("Building target subgraph", target=str(target))

    stacks = context.stacks
    target_stacks = [stacks[sid].get() for sid in target.stack_ids]
    by_filter = group_stacks_by_filter(target_stacks)

    graph_update_fn = partial(
        store_workflow_graph, ("target", target.stable_hash, config.quality.value)
    )
    node_status_fn = store_node_status

    graph = Graph(
        graph_update_fn=graph_update_fn, node_status_fn=node_status_fn
    )

    mock = config.development_settings.mock_pixinsight
    pp_fn = get_producer(MediaType.IMAGE, Target, QualityLevel.POST_PROCESSED, config.implementation)
    graph[(pp_fn, target, mock)] = {
        (build_target_channel_subgraph, target, filter_, tuple(channel_stacks), config)
        for filter_, channel_stacks in by_filter.items()
    }

    return graph


@node
def process_targets_for_ui(targets: dict) -> dict:
    def my_transform(v):
        if type(v) is np.ndarray:
            return v.tolist()
        if type(v) is SingleSphericalPolygon:
            return v.points.tolist()
        return v

    def prepare_target(target: Target):
        return convert_to_dict(target, transform=my_transform)

    # TODO: Parallelize this.
    from cache.context_ref import ContextRef
    ui_targets = {
        h: prepare_target(ref.get() if isinstance(ref, ContextRef) else ref)
        for h, ref in targets.items()
    }
    return ui_targets
