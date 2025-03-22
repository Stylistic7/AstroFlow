import ray

from dependency_graph import Graph
from targets.types import Target


@ray.remote
def build_target_subgraph(target: Target) -> Graph:
  print(f'### build_target_subgraph target {target}')
  # targets = context.targets
  # target = targets[target_id]

  # graph = Graph()
  # graph[(post_process_target, target_id)] = {
  #     (build_stack_subgraph, stack_id) for stack_id in target.stack_ids
  # }

  # return graph
