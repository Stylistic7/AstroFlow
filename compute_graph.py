import ray
import itertools
import os

from context import context
from image_metadata.image_metadata import scatter_compute_image_metadata
from spatial_association import find_components_by_cosine_similarity
from stack_identification import identify_stacks
from target_identification import group_stacks_into_targets
from calibration_frames import identify_calibration_stacks, associate_with_stacks
from image_previews import scatter_stack_thumbnails
from stacking import scatter_stack_subframes
from mosaics.identification import identify_mosiacs
from mosaics.viewbox import compute_mosaic_viewbox
from mosaics.merge_ops import new_make_merge_ops, merge_images
from dependency_graph import Graph, make_graph_from_list, print_graph
from post_process_target import post_process_target
from post_process_stack import post_process_stack
from post_process_mosaic import post_process_mosaic
from suggestions import make_suggestions
# from step_mosaic import merge_mosaic, build_mosaic_merge_graph
from stacking import stack_files
from post_process_stack import post_process_stack_artifacts

# @ray.remote
# def gather():
#   pass


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
def build_preview_graph():
  return make_graph_from_list([
      (scatter_stack_thumbnails, 'subframe'),
      # (build_mosaic_merge_graph, 'subframe'),
  ])


# TODO: Many stacks share dependencies. Create a graph such that these
# dependencies are represented by shared nodes in a graph, then remove the
# gross check for existing master files in stacking.py.
def build_stack_subgraph(stack_id):
  stacks = context.stacks
  stack = stacks[stack_id]

  graph = {}
  graph[(stack_files,
         stack_id)] = {(stack_files, d)
                       for substack in stack.contents
                       for d in stacks[substack].dependencies.values()
                       if stacks[d].image_type != 'light'}
  graph[(post_process_stack, stack_id)] = {(stack_files, stack_id)}

  return graph


def build_target_subgraph(target_id):
  print(f'### build_target_subgraph target_id {target_id}')
  targets = context.targets
  target = targets[target_id]

  graph = {}
  graph[(post_process_target, target_id)] = {
      (post_process_stack, stack_id) for stack_id in target.stack_ids
  }

  return graph

  # TODO: This seems ugly.
  # @ray.remote
  # def flatten_lists(results):
  #   result = list(itertools.chain.from_iterable(results))
  #   return result


# def build_merge_ops_subgraph(mode, mosaic_id, merge_ops):
#   graph = {}
#   for merge_op in merge_ops:
#     if merge_images.should_run(mode, mosaic_id, merge_op):
#       print(f'### merging_images {merge_op}')
#       graph[(merge_images, mode, merge_op)] = [
#           (post_process_stack, a.stack_id) for a in merge_op.input_artifacts
#       ]

#   return graph


def build_mosaic_filter_subgraph(mosaic_id, filter):
  print(f'### build_mosaic_filter_subgraph '
        f'mosaic_id {mosaic_id} filter {filter}')
  stacks = context.stacks
  targets = context.targets
  mosaics = context.mosaics
  mosaic = mosaics[mosaic_id]

  # TODO: What to do if there are multiple stacks for a filter? That's not
  # really supposed to happen.
  panels = {}
  for tile_indices, target_id in zip(mosaic.tile_indices, mosaic.target_ids):
    stacks_for_filter = [
        s for s in targets[target_id].stack_ids if stacks[s].filter == filter
    ]
    if not stacks_for_filter:
      continue

    assert len(stacks_for_filter) == 1
    stack_id = next(iter(stacks_for_filter))

    artifacts = post_process_stack_artifacts(stack_id)
    assert len(artifacts) == 1
    artifact = next(iter(artifacts))
    artifact.stack_id = stack_id

    panels[tile_indices] = artifact

  print(f'### found panels for filter {filter}: {panels}')
  merge_ops = new_make_merge_ops(panels)
  print(f'### merge_ops {merge_ops}')
  return merge_ops


def build_mosaic_subgraph(mosaic_id):
  print(f'### build_mosaic_subgraph mosaic_id {mosaic_id}')
  stacks = context.stacks
  targets = context.targets
  mosaics = context.mosaics
  mosaic = mosaics[mosaic_id]

  if not post_process_mosaic.should_run('master', mosaic_id):
    return Graph({})

  # Iterate over the filters in the mosaic.
  #   Get the stack_ids for all the panels in the mosaic for that filter.
  #   Get the artifact filenames for all the stacks.
  #   Compute the merge ops with those artifact filenames.
  #   Iterate over the merge ops.
  #     See if the merge op needs to run.
  #     If any merge op needs to run, keep all of the subsequent merge ops.
  #
  # Iterate over the merge ops that need to run.
  #   See if the post_process_stack operation needs to run for the stacks.
  #
  # Iterate over the post_process_stack operations that need to run.
  #   See if the stack_files operation needs to run.

  # TODO: Iterate over the actual filters in the mosaic.
  graph = {}
  for filter in ['H', 'S', 'O', 'L', 'R', 'G', 'B']:
    graph |= build_mosaic_filter_subgraph(mosaic_id, filter)

    # panels = {
    #     idx: get_stack(t)
    #     for idx, t in zip(mosaic.tile_indices, mosaic.target_ids)
    # }
    # merge_ops = new_make_merge_ops(panels)

    # graph = {}
    # merge_ops = [
    #     m for m in merge_ops if merge_images.should_run('master', mosaic_id, m)
    # ]

    # graph[(post_process_mosaic, 'master',
    #        mosaic_id)] = [(merge_images, 'master', m) for m in merge_ops]

  # graph[(post_process_mosaic, 'master', mosaic_id)] = (merge_mosaic, 'master')

  # viewbox_task = (compute_mosaic_viewbox, mosaic_id, 'master')
  # for filter, stack_ids in _stacks_for_filters(mosaic_id).items():
  #   for stack_id in stack_ids:
  #     graph[(place_panel,
  #            stack_id)] = [viewbox_task, (post_process_stack, stack_id)]

  # output_dir = os.path.join(context.output_dir, 'Mosaics', mosaic.name)
  # os.makedirs(output_dir, exist_ok=True)
  # graph |= _make_merge_ops(mosaic_id, output_dir)

  return Graph(graph)


def _stacks_for_filters(mosaic_id):
  mosaics = context.mosaics
  mosaic = mosaics[mosaic_id]
  targets = context.targets
  stacks = context.stacks

  def stacks_for_filter(filter):
    return [
        s for t in mosaic.target_ids for s in targets[t].stack_ids
        if stacks[s].filter == filter
    ]

  filters = ['H', 'S', 'O', 'L', 'R', 'G', 'B']
  result = {f: stacks_for_filter(f) for f in filters}
  return {k: v for k, v in result.items() if v}


# TODO: This places an image into a mosaic viewbox.
def place_panel(panel):
  pass


def _make_merge_ops(mosaic_id, output_dir):
  return {}


def build_main_processing_graph():
  shopping_cart = context.shopping_cart or []
  print(f'### build_main_processing_graph shopping_cart {shopping_cart}')

  # TODO: Maybe add prioritization options in the UI.
  graph = {}
  for type, id in shopping_cart:
    if type == 'stack':
      graph |= build_stack_subgraph(id)
    elif type == 'target':
      graph |= build_target_subgraph(id)
    if type == 'mosaic':
      graph |= build_mosaic_subgraph(id)

  print(f'### Finished composing main processing graph')
  print_graph(graph)

  return Graph(graph)


def build_compute_graph():
  return make_graph_from_list([
      (scatter_compute_image_metadata, 'input'),
      maybe_build_regroup_graph,
      make_suggestions,
      build_preview_graph,
      # This is the heavyweight graph that runs PixInsight and may take days.
      # build_main_processing_graph,
      # (scatter_compute_image_metadata, 'master'),
      # TODO: This is very slow.
      # (scatter_stack_thumbnails, 'master'),
      # (build_mosaic_merge_graph, 'master')
  ])


# TODO: Add a cleanup task that deletes unreferenced stuff from the cache.
# Can I use some kind of automatic ref-counting?
