import ray
import numpy as np
from spherical_geometry.polygon import SingleSphericalPolygon

from context import context
from dependency_graph import Graph
from targets.types import Target
from targets.post_process import post_process_target
from targets.types import TargetsDict
from stacks.compute_graph import build_stacking_subgraph
from stacks.convert_to_dict import convert_to_dict
from dependency_graph import run_graph
from previews.types import QualityLevel
from dependency_graph.workflow_graph_adapter import to_workflow_graph


@ray.remote
def build_target_subgraph(mode: QualityLevel, target: Target) -> Graph:
    print(f"### build_target_subgraph target {target}")

    stacks = context.stacks

    graph = Graph()
    graph[(post_process_target, target)] = {
        (build_stacking_subgraph, mode, stacks[stack_id])
        for stack_id in target.stack_ids
    }

    return graph


@ray.remote
def process_targets_for_ui(targets: TargetsDict) -> None:
    def my_transform(v):
        if type(v) is np.ndarray:
            return v.tolist()
        if type(v) is SingleSphericalPolygon:
            return v.points.tolist()
        return v

    def prepare_target(target: Target):
        result = convert_to_dict(target, transform=my_transform)

        graph_status = None

        def graph_update_fn(g):
            nonlocal graph_status
            graph_status = g

        graph = ray.get(build_target_subgraph.remote(QualityLevel.MASTER, target))

        graph_results, _ = run_graph(
            graph, graph_update_fn=graph_update_fn, dry_run=True
        )

        result["contains_exception"] = graph_results.contains_exception()
        result["is_complete"] = graph_results.is_complete()
        result["workflow_graph"] = to_workflow_graph(graph_status)

        return result

    # TODO: Parallelize this.
    ui_targets = {h: prepare_target(s) for h, s in targets.items()}
    context.ui_targets = ui_targets
    return ui_targets
