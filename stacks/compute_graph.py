import ray

from context import context
from stacks.stacking import stack_files
from stacks.post_process_stack import post_process_stack
from dependency_graph import Graph


@ray.remote
def build_stack_subgraph(stack_id):
  stacks = context.stacks
  stack = stacks[stack_id]

  graph = Graph()
  graph[(stack_files,
         stack_id)] = {(stack_files, d)
                       for substack in stack.contents
                       for d in stacks[substack].dependencies.values()
                       if stacks[d].image_type != 'light'}
  graph[(post_process_stack, stack_id)] = {(stack_files, stack_id)}

  return graph
