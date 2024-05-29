import ray

from context import context
from image_metadata import scatter_compute_image_metadata
from spatial_association import find_components_by_cosine_similarity
from stack_identification import identify_stacks
from target_identification import group_stacks_into_targets
from calibration_frames import identify_calibration_stacks, associate_with_stacks
from image_previews import scatter_stack_thumbnails
from stacking import scatter_stack_subframes
from step_mosaic import scatter_merge_mosaics
from mosaics.identification import identify_mosiacs
from dependency_graph import Graph, make_graph_from_list
from process_target import gather_stacks


@ray.remote
def maybe_build_regroup_graph():
  if context.regroup:
    return make_graph_from_list([
        find_components_by_cosine_similarity, identify_stacks,
        identify_calibration_stacks, associate_with_stacks,
        group_stacks_into_targets, identify_mosiacs
    ])
  else:
    return {}


def build_item_processing_graph(item):
  print(f'### running compute graph for item {item}')
  return Graph({})


@ray.remote
def build_processing_graph():
  stacks = context.stacks
  targets = context.targets
  shopping_cart = context.shopping_cart or []

  # TODO: Maybe add prioritization options in the UI.
  # TODO: Remove duplicates.
  stack_nodes = []
  for type, id in shopping_cart:
    if type == 'stack':
      stack = stacks[id]
      print(f'### stack in shopping cart: {stack}')
      stack_nodes.append((scatter_stack_subframes, id))

    if type == 'target':
      target = targets[id]
      print(f'### target in shopping cart: {target}')
      for stack_id in target.stack_ids:
        stack_nodes.append((scatter_stack_subframes, stack_id))

  graph = {}
  graph[gather_stacks] = stack_nodes
  return Graph(graph)


def build_compute_graph():
  return make_graph_from_list([(scatter_compute_image_metadata, 'all'),
                               maybe_build_regroup_graph,
                               (scatter_stack_thumbnails, 'subframe'),
                               (scatter_merge_mosaics, 'subframe'),
                               build_processing_graph,
                               (scatter_compute_image_metadata, 'master'),
                               (scatter_stack_thumbnails, 'master'),
                               (scatter_merge_mosaics, 'master')])


# TODO: Add a cleanup task that deletes unreferenced stuff from the cache.
# Can I use some kind of automatic ref-counting?
