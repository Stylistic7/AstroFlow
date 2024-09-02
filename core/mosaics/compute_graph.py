import ray
import os
import numpy as np

from artifacts import Artifact, CacheArtifact, FileArtifact, read_artifact_image, write_artifact, artifact_exists
from context import context
from dependency_graph import Graph, print_graph, make_graph_from_list
from previews import array_to_png
# TODO: Unify quarantining
import quarantining
from mosaics.classes import Panel
from mosaics.merge_ops import make_merge_ops
from mosaics.viewbox import compute_mosaic_viewbox
from mosaics.placement import mosaic_by_coordinates, mosaic_by_coordinates_artifacts
from stacks.post_process import post_process_stack_artifacts
from stacks.compute_graph import build_stack_subgraph
from artifacts.decorators import saves_artifacts
from previews import resize_array
from metadata.actor import get_metadata_for_mosaic


@ray.remote
def done_stacking():
  pass


@ray.remote
def build_stacks_for_mosaic_subgraph(mode, mosaic_id):
  print(f'### build_stacks_for_mosaic_subgraph {mode} {mosaic_id}')
  assert mode == 'master'

  mosaics = context.mosaics
  targets = context.targets

  graph = Graph()
  graph[done_stacking] = {(build_stack_subgraph, stack_id)
                          for target_id in mosaics[mosaic_id].target_ids
                          for stack_id in targets[target_id].stack_ids}

  return graph


# TODO: Make sure these are parallelized over all the mosaics.
@ray.remote
def build_mosaic_filter_subgraphs(mode, mosaic_id):
  print(f'### build_mosaic_filter_subgraphs mode {mode} mosaic_id {mosaic_id}')

  # TODO: Run all of the (build_stack_subgraph, stack_id) tasks first.

  # Get all of the image metadata for the entire mosaic (which may involve
  # thousands of individual subframes) all at once. Pass it around as an
  # ObjectRef inside a tuple, which is a Ray convention that prevents it from
  # being dereferenced until desired.
  metadata_ref = (ray.put(get_metadata_for_mosaic(mosaic_id)), 0)

  # TODO: If the final output artifact exists, there's no reason to build the
  # graph.
  graph = Graph()
  viewbox_task = (compute_mosaic_viewbox, mode, mosaic_id)
  graph[viewbox_task] = []

  # TODO: Iterate over the actual filters in the mosaic.
  for filter in ['H', 'S', 'O', 'L', 'R', 'G', 'B']:
    graph |= ray.get(
        build_mosaic_filter_subgraph.remote(mode, mosaic_id, filter,
                                            viewbox_task, metadata_ref))

  # TODO: Run post-processing here.

  return graph


@ray.remote
def build_mosaic_subgraph(mode, mosaic_id):
  if mode == 'subframe':
    return ray.get(build_mosaic_filter_subgraphs.remote(mode, mosaic_id))
  elif mode == 'master':
    stack_subgraph = (build_stacks_for_mosaic_subgraph, mode, mosaic_id)
    mosaic_subgraph = (build_mosaic_filter_subgraphs, mode, mosaic_id)
    return make_graph_from_list([stack_subgraph, mosaic_subgraph])


def _get_best_light_frame(stack_id, metadata):
  stack = context.stacks[stack_id]
  light_frames_by_stars = quarantining.get_light_frames_sorted_by_stars(
      stack.all_frames(), metadata)

  return light_frames_by_stars[0]


@ray.remote
def build_mosaic_filter_subgraph(mode, mosaic_id, filter, viewbox_task,
                                 metadata_ref):
  print(f'### build_mosaic_filter_subgraph mode {mode} '
        f'mosaic_id {mosaic_id} filter {filter}')
  stacks = context.stacks
  targets = context.targets
  mosaics = context.mosaics
  mosaic = mosaics[mosaic_id]

  metadata = ray.get(metadata_ref[0])

  graph = Graph()

  # TODO: What to do if there are multiple stacks for a filter? That's not
  # really supposed to happen.
  panels = {}
  from_task = {}
  for tile_indices, target_id in zip(mosaic.tile_indices, mosaic.target_ids):
    stacks_for_filter = [
        s for s in targets[target_id].stack_ids if stacks[s].filter == filter
    ]
    if not stacks_for_filter:
      continue

    if len(stacks_for_filter) > 1:
      print(f'### WARNING: Multiple stacks for filter {filter} '
            f'for target {targets[target_id]}')
    stack_id = next(iter(stacks_for_filter))

    # TODO: Find a way to streamline all these if-else statements in general.
    if mode == 'subframe':
      best_light_frame = _get_best_light_frame(stack_id, metadata)
      input_artifact = FileArtifact(best_light_frame.filename,
                                    best_light_frame.content_hash)
    elif mode == 'master':
      input_artifact = post_process_stack_artifacts(stack_id)[0]

    placement_task = (mosaic_by_coordinates, mode, mosaic_id, filter, stack_id,
                      input_artifact, metadata_ref)
    graph[placement_task] = {viewbox_task}

    placed_artifact = mosaic_by_coordinates_artifacts(mode, mosaic_id, filter,
                                                      stack_id,
                                                      input_artifact)[0]
    print(f'### build_mosaic_filter_subgraph placed_artifact '
          f'is {placed_artifact}')
    panel = Panel(tile_indices=tile_indices,
                  content_hash=placed_artifact.content_hash,
                  input_artifact=placed_artifact)

    from_task[panel] = placement_task
    panels[tile_indices] = panel

  if len(panels) == 0:
    print(f'### no panels found for filter {filter}')
    return {}

  print(f'### found panels for filter {filter}:')
  for tile_indices, panel in panels.items():
    print(f'### tile_indices {tile_indices} => panel {panel}')

  _, merge_graph = make_merge_ops(mode, mosaic_id, filter, panels, from_task)
  print(f'### make_merge_ops returned graph:')
  print_graph(merge_graph)

  if merge_graph:
    print(f'### merge_graph has length {len(merge_graph)}')
    graph |= merge_graph
    graph[(
        merge_graph_done,
        mode,
        mosaic_id,
        filter,
    )] = {list(merge_graph.keys())[-1]}

  return graph


def merge_graph_done_artifacts(mode,
                               mosaic_id,
                               filter,
                               *,
                               ancestor_results=None):
  mosaic = context.mosaics[mosaic_id]
  key = ('mosaic_thumbnail_image', mode, mosaic_id, filter)

  if mode == 'subframe':
    return [CacheArtifact(key, mosaic.content_hash)]
  elif mode == 'master':
    # In master mode, we save both the real high-res FITS file on disk, but
    # also a thumbnail image in the cache.
    filename = os.path.join(context.output_dir, 'Mosaics', mosaic.name,
                            f'final_merged_{mosaic_id}_{filter}.fit')
    return [
        FileArtifact(filename, mosaic.content_hash),
        CacheArtifact(key, mosaic.content_hash)
    ]


# TODO: Pass the output artifact to the make_merge_ops() function and eliminate
# this function.
@ray.remote
@saves_artifacts(merge_graph_done_artifacts)
def merge_graph_done(mode, mosaic_id, filter, *, ancestor_results=None):
  mosaic = context.mosaics[mosaic_id]

  assert type(ancestor_results) is list
  assert len(ancestor_results) == 1
  final_artifact = ancestor_results[0][0]
  assert isinstance(final_artifact,
                    Artifact), f'final_artifact has type {type(final_artifact)}'
  print(f'### Merging {mosaic} mode {mode} is done, '
        f'final_artifact is {final_artifact}')

  output_artifacts = merge_graph_done_artifacts(mode, mosaic_id, filter)

  # TODO: This stuff should be standardized and put into the @saves_artifacts
  # decorator.
  if not artifact_exists(final_artifact):
    if mode == 'subframe':
      thumbnail_artifact, = output_artifacts
      return [thumbnail_artifact]
    elif mode == 'master':
      file_artifact, thumbnail_artifact = output_artifacts
      return [file_artifact, thumbnail_artifact]

  image = read_artifact_image(final_artifact)
  assert not np.isnan(np.min(image))

  header = {'IMAGETYP': 'FinalMosaic Light'}

  # TODO: Thumbnails should always be stored as autostretched PNG, for speed.
  if mode == 'subframe':
    thumbnail_artifact, = output_artifacts
    write_artifact(thumbnail_artifact, image, header)
    return [thumbnail_artifact]
  elif mode == 'master':
    file_artifact, thumbnail_artifact = output_artifacts
    write_artifact(file_artifact, image, header)
    image = resize_array(image, context.master_thumbnail_width)
    assert not np.isnan(np.min(image))
    write_artifact(thumbnail_artifact, image, header)
    return [file_artifact, thumbnail_artifact]
