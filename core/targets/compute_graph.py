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
from targets.post_process import post_process_target
from targets.types import TargetsDict
from stacks.compute_graph import build_stacking_subgraph
from stacks.convert_to_dict import convert_to_dict
from decisions.settings import DevelopmentSettings
from dependency_graph.workflow_graph_adapter import (
    store_workflow_graph,
    store_node_status,
)

log = structlog.get_logger()


@ray.remote(num_cpus=0.1)
@beartype
def build_target_subgraph(
    target: Target,
    config: "ProcessingConfig",
) -> Graph:
    from decisions.settings import ProcessingConfig
    from frozendict import frozendict as fd
    log.info("Building target subgraph", target=str(target))

    stacks = context.stacks
    all_decisions = context.settings.decision_overrides

    def config_for(stack_id):
        decisions = fd({k: v for k, v in all_decisions.items() if len(k) >= 3 and k[2] == stack_id})
        return ProcessingConfig(
            calibration_stacks=config.calibration_stacks,
            development_settings=config.development_settings,
            min_frames_in_stack=config.min_frames_in_stack,
            decisions=decisions,
        )

    quality = "DownscaledMaster" if config.downscale_factor > 1 else "Master"
    graph_update_fn = partial(
        store_workflow_graph, ("target", target.stable_hash, quality)
    )
    node_status_fn = store_node_status

    graph = Graph(
        graph_update_fn=graph_update_fn, node_status_fn=node_status_fn
    )

    mock = config.development_settings.mock_pixinsight
    graph[(post_process_target, target, stacks, mock)] = {
        (build_stacking_subgraph, stacks[stack_id].get(), config_for(stack_id))
        for stack_id in target.stack_ids
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
