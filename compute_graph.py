import ray
import itertools

from context import context
from image_metadata import scatter_compute_image_metadata, compute_image_metadata_for_filenames
from spatial_association import find_components_by_cosine_similarity
from stack_identification import identify_stacks
from target_identification import group_stacks_into_targets
from calibration_frames import identify_calibration_stacks, associate_with_stacks
from image_previews import scatter_stack_thumbnails
from stacking import scatter_stack_subframes
from step_mosaic import scatter_merge_mosaics
from mosaics.identification import identify_mosiacs
from dependency_graph import Graph, Node, make_graph_from_list
from post_process_target import post_process_target
from post_process_stack import post_process_stack
from post_process_mosaic import post_process_mosaic
from suggestions import make_suggestions
from step_mosaic import merge_mosaic


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
  # The stacked master filenames are passed along directly here.
  return Graph({(post_process_stack, id): {(scatter_stack_subframes, id)}})


@ray.remote
def build_target_subgraph(id):
  target = context.targets[id]
  print(f'### build_target_subgraph for target {target}')
  # The post-processed filenames are passed along directly here.
  return Graph({
      (post_process_target, id): [
          (build_stack_subgraph, s) for s in target.stack_ids
      ]
  })


# TODO: This seems ugly.
@ray.remote
def flatten_lists(results):
  result = list(itertools.chain.from_iterable(results))
  print('### flatten_lists returning', result)
  return result


@ray.remote
def build_mosaic_subgraph(id):
  mosaic = context.mosaics[id]
  print(f'### build_mosaic_subgraph for mosaic {mosaic}')
  # The masters need to be plate solved, etc. before mosaic merge, so there's
  # an explicit task here to get their metadata.
  merge_task = (merge_mosaic, id, 'master')
  return Graph({
      (post_process_mosaic, id): {merge_task},
      merge_task: {compute_image_metadata_for_filenames},
      compute_image_metadata_for_filenames: {flatten_lists},
      flatten_lists: [(build_target_subgraph, t) for t in mosaic.target_ids]
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
    elif type == 'mosaic':
      subgraphs.append((build_mosaic_subgraph, id))

  return Graph({gather_processing_graph: subgraphs})


def build_compute_graph():
  return make_graph_from_list([
      (scatter_compute_image_metadata, 'all'),
      maybe_build_regroup_graph,
      make_suggestions,
      # Compute preview images from subframes (before stacking).
      (scatter_stack_thumbnails, 'subframe'),
      (scatter_merge_mosaics, 'subframe'),
      # This is the heavyweight graph that runs PixInsight and may take days.
      build_processing_graph,
      # Compute previews from masters (after stacking).
      (scatter_compute_image_metadata, 'master'),
      (scatter_stack_thumbnails, 'master'),
      # (scatter_merge_mosaics, 'master')
  ])


# TODO: Add a cleanup task that deletes unreferenced stuff from the cache.
# Can I use some kind of automatic ref-counting?
