import ray

from context import context
from stacks.stacking import stack_files
from stacks.post_process_stack import post_process_stack
from dependency_graph import Graph, Node
from metadata.image_metadata import compute_image_metadata
from stacks.identification import identify_stacks


@ray.remote
def gather_identify_stacks(ancestor_results=None):
  pass


@ray.remote
def scatter_identify_stacks():
  graph = Graph({
      gather_identify_stacks: {
          (identify_stacks, g) for g in context.spatial_groups
      }
  })

  return graph


# TODO: Should these build_*_subgraph functions eliminate nodes that don't
# need to run? If the final post-processed stack artifact exists, there's
# no reason to check all of the predecing artifacts.
@ray.remote
def build_stack_subgraph(stack_id):
  stacks = context.stacks
  stack = stacks[stack_id]

  stack_task = (stack_files, stack_id)
  metadata_task = Node(compute_image_metadata, name=stack_id)
  post_process_task = (post_process_stack, stack_id)

  graph = Graph()
  graph[stack_task] = {(stack_files, d)
                       for substack in stack.contents
                       for d in stacks[substack].dependencies.values()
                       if stacks[d].image_type != 'light'}
  graph[metadata_task] = {stack_task}
  graph[post_process_task] = {metadata_task}

  return graph
