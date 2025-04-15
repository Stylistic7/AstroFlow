import ray

from context import context
from dependency_graph import Graph
from targets.types import Target
from targets.post_process import post_process_target
from stacks.compute_graph import build_stacking_subgraph


@ray.remote
def build_target_subgraph(target: Target) -> Graph:
  print(f'### build_target_subgraph target {target}')

  stacks = context.stacks

  graph = Graph()
  graph[(post_process_target, target)] = {(build_stacking_subgraph,
                                           stacks[stack_id])
                                          for stack_id in target.stack_ids}

  return graph
