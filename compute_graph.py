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
from post_process_target import post_process_target
from post_process_stack import post_process_stack


@ray.remote
def gather_processing_graph():
  pass


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


@ray.remote
def build_stack_subgraph(id):
  stack = context.stacks[id]
  print(f'### build_stack_subgraph for stack id {stack}')
  return Graph({(post_process_stack, id): {(scatter_stack_subframes, id)}})


@ray.remote
def build_target_subgraph(id):
  target = context.targets[id]
  print(f'### build_target_subgraph for target {target}')
  return Graph({
      (post_process_target, id): [
          (post_process_stack, s) for s in target.stack_ids
      ]
  })


@ray.remote
def build_processing_graph():
  shopping_cart = context.shopping_cart or []
  print(f'### shopping_cart is {shopping_cart}')

  # TODO: Maybe add prioritization options in the UI.
  # TODO: Remove duplicates.
  subgraphs = []
  for type, id in shopping_cart:
    print(f'### item in shopping_cart: {type} {id}')
    if type == 'stack':
      subgraphs.append((build_stack_subgraph, id))
    elif type == 'target':
      subgraphs.append((build_target_subgraph, id))

  return Graph({gather_processing_graph: subgraphs})


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
