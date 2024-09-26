import ray

from context import context
from stacks.stacking import stack_files
from stacks.post_process import post_process_stack
from dependency_graph import Graph, Node
from metadata.image_metadata import compute_image_metadata, gather_compute_image_metadata


# TODO: Should these build_*_subgraph functions eliminate nodes that don't
# need to run? If the final post-processed stack artifact exists, there's
# no reason to check all of the predecing artifacts.
@ray.remote
def build_stack_subgraph(stack_id):
  stacks = context.stacks
  stack = stacks[stack_id]

  stack_task = (stack_files, stack_id)
  stacking_metadata_task = Node(compute_image_metadata,
                                name=f'stacking_{stack_id}')
  post_process_task = (post_process_stack, stack_id)
  post_process_metadata_task = Node(compute_image_metadata,
                                    name=f'post_process_{stack_id}')

  graph = Graph()
  graph[stack_task] = {(stack_files, d)
                       for substack in stack.contents.values()
                       for d in substack.dependencies.values()
                       if stacks[d].image_type != 'light'}
  graph[stacking_metadata_task] = {stack_task}
  graph[post_process_task] = {stacking_metadata_task}
  graph[post_process_metadata_task] = {post_process_task}
  graph[(gather_compute_image_metadata,)] = {post_process_metadata_task}

  return graph
