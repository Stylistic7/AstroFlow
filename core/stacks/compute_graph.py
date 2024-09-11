import ray

from context import context
from stacks.stacking import stack_files
from stacks.post_process import post_process_stack
from dependency_graph import Graph, Node
from metadata.image_metadata import compute_image_metadata
from stacks.identification import identify_stacks
from targets.identification import group_stacks_into_targets


@ray.remote
def gather_identify_stacks(ancestor_results=None):
  pass


# TODO: This could probably be simplified.
#
# TODO: The spatial groups should include metadata refs, and they should
# be passed down into identify_stacks().
#
# TODO: The stack identification and target identification should be separated
# for clarity.
@ray.remote
def scatter_identify_stacks():
  graph = Graph()
  target_tasks = []
  for g in context.spatial_groups:
    target_task = (group_stacks_into_targets, g)
    target_tasks.append(target_task)
    graph[target_task] = {(identify_stacks, g)}

  graph[gather_identify_stacks] = target_tasks

  return graph


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
                       for substack in stack.contents
                       for d in stacks[substack].dependencies.values()
                       if stacks[d].image_type != 'light'}
  graph[stacking_metadata_task] = {stack_task}
  graph[post_process_task] = {stacking_metadata_task}
  graph[post_process_metadata_task] = {post_process_task}

  return graph
