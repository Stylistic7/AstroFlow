import ray
import numpy as np
from beartype import beartype
from spherical_geometry.polygon import SingleSphericalPolygon
from functools import partial

from context import context
from decorators import cached
from dependency_graph import Graph
from targets.types import Target
from targets.post_process import post_process_target
from targets.types import TargetsDict
from stacks.compute_graph import build_stacking_subgraph
from stacks.convert_to_dict import convert_to_dict
from customization.settings import DevelopmentSettings
from dependency_graph.workflow_graph_adapter import (
    store_workflow_graph,
    store_node_status,
)


@ray.remote
@beartype
def build_target_subgraph(
    target: Target,
    development_settings: DevelopmentSettings,
    min_frames_in_stack: int,
) -> Graph:
    print(f"### build_target_subgraph target {target}")

    stacks = context.stacks
    calibration_stacks = context.calibration_stacks

    graph_update_fn = partial(
        store_workflow_graph, ("target", target.stable_hash)
    )
    node_status_fn = store_node_status

    graph = Graph(
        graph_update_fn=graph_update_fn, node_status_fn=node_status_fn
    )

    graph[(post_process_target, target, stacks)] = {
        (build_stacking_subgraph, stacks[stack_id], calibration_stacks, development_settings, min_frames_in_stack)
        for stack_id in target.stack_ids
    }

    return graph


@cached.remote
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
    ui_targets = {h: prepare_target(s) for h, s in targets.items()}
    return ui_targets
