import ray
import os
import numpy as np
from typing import Any

from artifacts import Artifact, CacheArtifact, FileArtifact, read_artifact_image, write_artifact, artifact_exists
from context import context
from dependency_graph import Graph, Node, TaskDescriptor, print_graph, make_graph_from_list
# TODO: Unify quarantining
import quarantining
from metadata.classes import ImageMetadata, MetadataDict
from mosaics.types import MosaicHash
from mosaics.classes import Panel, Viewbox
from mosaics.merge_images import merge_images
from mosaics.mosaic import Mosaic
from mosaics.merge_ops import make_merge_ops
from mosaics.viewbox import compute_mosaic_viewbox
from mosaics.placement import place_image
from mosaics.types import TileIndices
from stacks.compute_graph import build_stack_subgraph
from stacks.types import StackHash
from stacks.composite_stack import CompositeStack
from targets.types import TargetHash
from artifacts.decorators import saves_artifacts
from previews import resize_array, QualityLevel
from profiling import track_performance
from dependency_graph.astroflow_node import AstroFlowNode
from targets import Target


@ray.remote
def done_stacking() -> None:
  pass


# TODO: This should depend on target subgraphs, which will in turn depend on
# stack subgraphs.
# @ray.remote
# def build_stacks_for_mosaic_subgraph(mode: QualityLevel,
#                                      mosaic_id: MosaicHash) -> Graph:
#   print(f'### build_stacks_for_mosaic_subgraph {mode} {mosaic_id}')
#   assert mode == 'master'

#   mosaics = context.mosaics
#   targets = context.targets

#   graph = Graph()
#   graph[done_stacking] = {(build_stack_subgraph, stack_id)
#                           for target_id in mosaics[mosaic_id].target_ids
#                           for stack_id in targets[target_id].stack_ids}

#   return graph

# TODO: Make sure these are parallelized over all the mosaics.
# @ray.remote
# def build_mosaic_filter_subgraphs(mode: QualityLevel,
#                                   mosaic_id: MosaicHash) -> Graph:
#   print(f'### build_mosaic_filter_subgraphs mode {mode} mosaic_id {mosaic_id}')

#   # TODO: Run all of the (build_stack_subgraph, stack_id) tasks first.
#   #
#   # TODO: If the final output artifact exists, there's no reason to build the
#   # graph.
#   graph = Graph()
#   viewbox_task = (compute_mosaic_viewbox, mosaic_id)
#   graph[viewbox_task] = []

#   # TODO: Iterate over the actual filters in the mosaic.
#   for filter in ['H', 'S', 'O', 'L', 'R', 'G', 'B']:
#     subgraph = ray.get(
#         build_mosaic_filter_subgraph.remote(mode, mosaic_id, filter,
#                                             viewbox_task))
#     graph |= subgraph

#   # TODO: Run post-processing here.

#   return graph

# @ray.remote
# def build_mosaic_subgraph(mode: QualityLevel, mosaic_id: MosaicHash) -> Graph:
#   if mode == QualityLevel.SUBFRAME:
#     return ray.get(build_mosaic_filter_subgraphs.remote(mode, mosaic_id))
#   elif mode == QualityLevel.MASTER:
#     stack_subgraph = (build_stacks_for_mosaic_subgraph, mode, mosaic_id)
#     mosaic_subgraph = (build_mosaic_filter_subgraphs, mode, mosaic_id)
#     return make_graph_from_list([stack_subgraph, mosaic_subgraph])

#   raise ValueError(f'Unknown quality level {mode}')


@AstroFlowNode()
def mosaic_done() -> None:
  pass


@AstroFlowNode()
def build_mosaic_subgraph(mode: QualityLevel, mosaic: Mosaic) -> Graph:
  print('### build_mosaic_subgraph for mosaic {mosaic}')
  viewbox = compute_mosaic_viewbox(mosaic)

  graph = Graph()

  # TODO: Iterate over the actual filters in the mosaic.
  filters = ['H', 'S', 'O', 'L', 'R', 'G', 'B']
  graph[mosaic_done] = [(build_mosaic_filter_subgraph, mode, mosaic, filter,
                         viewbox) for filter in filters]

  return graph


def get_best_light_frame_for_stack(stack: CompositeStack,
                                   metadata: MetadataDict) -> ImageMetadata:

  all_frames = stack.all_frames()
  assert all_frames

  light_frames_by_stars = quarantining.get_light_frames_sorted_by_stars(
      all_frames, metadata)

  result = light_frames_by_stars[0]

  # print(f'### get_best_light_frame for stack {stack} returning {result}')
  return result


@AstroFlowNode()
def get_best_light_frame_for_target(target: Target,
                                    filter: str) -> FileArtifact | None:
  stacks = context.stacks
  metadata = context.metadata

  stacks_for_filter = [
      s for s in target.stack_ids if stacks[s].filter == filter
  ]
  if not stacks_for_filter:
    return None

  if len(stacks_for_filter) > 1:
    print(f'### WARNING: Multiple stacks for filter {filter} '
          f'for target {target}')
  stack_id = next(iter(stacks_for_filter))

  print(f'### Found stack_id {stack_id} for filter {filter}')

  best_light_frame = get_best_light_frame_for_stack(stacks[stack_id], metadata)
  return FileArtifact(best_light_frame.filename, best_light_frame.content_hash)


@AstroFlowNode()
def build_placement_subgraph(mode: QualityLevel, mosaic: Mosaic, target: Target,
                             filter: str, tile_indices: TileIndices,
                             viewbox: Viewbox):
  return make_graph_from_list([
      (get_best_light_frame_for_target, target, filter),
      (place_image, mode, mosaic, tile_indices, viewbox)
  ])


@AstroFlowNode()
def build_mosaic_filter_subgraph(mode: QualityLevel, mosaic: Mosaic,
                                 filter: str, viewbox: Viewbox) -> Graph:
  targets = context.targets

  print(f'### build_mosaic_filter_subgraph mode {mode} '
        f'mosaic {mosaic} filter {filter}')

  # For strip index 0:
  #   merge(place_a, place_b) -> merge_b
  #   merge(merge_b, place_c) -> merge_c
  #   merge(merge_c, place_d) -> merge_d
  #
  # For strip index 1:
  #   merge(place_e, place_f) -> merge_f
  #   merge(merge_f, place_g) -> merge_g
  #   merge(merge_g, place_h) -> merge_h
  #
  # For strip index 2:
  #   merge(place_i, place_j) -> merge_j
  #   merge(merge_j, place_k) -> merge_k
  #   merge(merge_k, place_l) -> merge_l
  graph = Graph()
  strip_tasks = []
  prev_task = None
  for strip_index, strip in enumerate(mosaic.strips):
    for i, target_id in enumerate(strip):
      if i == 0:
        prev_task = (build_placement_subgraph, mode, mosaic, targets[target_id],
                     filter, TileIndices((strip_index, i)), viewbox)
      else:
        merge_task = Node(merge_images, mode, name=f'merge_{target_id}')
        graph[merge_task] = [
            prev_task,
            (build_placement_subgraph, mode, mosaic, targets[target_id], filter,
             TileIndices((strip_index, i)), viewbox)
        ]
        prev_task = merge_task

    strip_tasks.append(prev_task)

  # For a mosaic with three strips:
  #   merge(merge_d, merge_h) -> merge_strips_1
  #   merge(merge_strips_1, merge_l) -> final
  for i, strip in enumerate(mosaic.strips):
    if i == 0:
      prev_task = Node(merge_images, mode, name=f'merge_{strip[-1]}')
    else:
      merge_task = Node(merge_images, mode, name=f'merge_strip_{i}')
      graph[merge_task] = [
          prev_task,
          Node(merge_images, mode, name=f'merge_{strip[-1]}')
      ]
      prev_task = merge_task

  return graph

  # TODO: What to do if there are multiple stacks for a filter? That's not
  # really supposed to happen.
  # panels: dict[TileIndices, Panel] = {}
  # from_task: dict[Panel, TaskDescriptor] = {}
  # placement_tasks = []
  # for tile_indices, target_id in zip(mosaic.tile_indices, mosaic.target_ids):

  #   placement_task = (place_image, mode, mosaic.stable_hash, tile_indices,
  #                     input_artifact, viewbox)
  #   graph[placement_task] = {(build_stack_subgraph, stack_id)}
  #   placement_tasks.append(placement_task)

  # merge_ops_task = (build_merge_graph, mode, mosaic, filter)
  # graph[merge_ops_task] = placement_tasks
  # graph[merge_graph_done] = {merge_ops_task}

  #   # TODO: Find a way to streamline all these if-else statements in general.
  #   input_artifact: Artifact
  #   if mode == QualityLevel.SUBFRAME:
  #     best_light_frame = get_best_light_frame(stack_id, metadata)
  #     input_artifact = FileArtifact(best_light_frame.filename,
  #                                   best_light_frame.content_hash)
  #   elif mode == QualityLevel.MASTER:
  #     input_artifact = post_process_stack_artifacts(stack_id)[0]

  #   placement_task = (place_image, mode, mosaic_id, filter, stack_id,
  #                     input_artifact)
  #   graph[placement_task] = {viewbox_task}

  #   placed_artifact = mosaic_by_coordinates_artifacts(mode, mosaic_id, filter,
  #                                                     stack_id,
  #                                                     input_artifact)[0]
  #   print(f'### build_mosaic_filter_subgraph placed_artifact '
  #         f'is {placed_artifact}')
  #   panel = Panel(tile_indices=tile_indices,
  #                 content_hash=placed_artifact.content_hash,
  #                 input_artifact=placed_artifact)

  #   from_task[panel] = placement_task
  #   panels[tile_indices] = panel

  # if len(panels) == 0:
  #   print(f'### no panels found for filter {filter}')
  #   return Graph({})

  # print(f'### found panels for filter {filter}:')
  # for tile_indices, panel in panels.items():
  #   print(f'### tile_indices {tile_indices} => panel {panel}')

  # _, merge_graph = make_merge_ops(mode, mosaic_id, filter, panels, from_task)

  # if merge_graph:
  #   print(f'### merge_graph has length {len(merge_graph)}')
  #   graph |= merge_graph
  #   graph[(
  #       merge_graph_done,
  #       mode,
  #       mosaic_id,
  #       filter,
  #   )] = {list(merge_graph.keys())[-1]}

  # return graph


def merge_graph_done_artifacts(mode: QualityLevel, mosaic_id: MosaicHash,
                               filter: str,
                               final_artifact: Artifact) -> list[Artifact]:
  mosaic = context.mosaics[mosaic_id]
  key = ('mosaic_thumbnail_image', mode, mosaic_id, filter)

  if mode == QualityLevel.SUBFRAME:
    return [CacheArtifact(key, mosaic.content_hash)]
  elif mode == QualityLevel.MASTER:
    # In master mode, we save both the real high-res FITS file on disk, but
    # also a thumbnail image in the cache.
    filename = os.path.join(context.output_dir, 'Mosaics', mosaic.name,
                            f'final_merged_{mosaic_id}_{filter}.fit')
    return [
        FileArtifact(filename, mosaic.content_hash),
        CacheArtifact(key, mosaic.content_hash)
    ]

  raise ValueError(f'Uknown quality level {mode}')


@ray.remote
def merge_graph_done() -> None:
  pass


# TODO: Pass the output artifact to the make_merge_ops() function and eliminate
# this function.
# @ray.remote
# @saves_artifacts(merge_graph_done_artifacts)
# def merge_graph_done(mode: QualityLevel, mosaic_id: MosaicHash, filter: str,
#                      final_artifact: Artifact) -> list[Artifact]:
#   mosaic = context.mosaics[mosaic_id]

#   assert isinstance(
#       final_artifact,
#       (FileArtifact,
#        CacheArtifact)), f'final_artifact has type {type(final_artifact)}'
#   print(f'### Merging {mosaic} mode {mode} is done, '
#         f'final_artifact is {final_artifact}')

#   output_artifacts = merge_graph_done_artifacts(mode, mosaic_id, filter,
#                                                 final_artifact)

#   # TODO: This stuff should be standardized and put into the @saves_artifacts
#   # decorator.
#   if not artifact_exists(final_artifact):
#     if mode == QualityLevel.SUBFRAME:
#       thumbnail_artifact, = output_artifacts
#       return [thumbnail_artifact]
#     elif mode == QualityLevel.MASTER:
#       file_artifact, thumbnail_artifact = output_artifacts
#       return [file_artifact, thumbnail_artifact]

#   image = read_artifact_image(final_artifact)
#   assert not np.isnan(np.min(image))

#   header = {'IMAGETYP': 'FinalMosaic Light'}

#   # TODO: Thumbnails should always be stored as autostretched PNG, for speed.
#   if mode == QualityLevel.SUBFRAME:
#     thumbnail_artifact, = output_artifacts
#     write_artifact(thumbnail_artifact, image, header)
#     return [thumbnail_artifact]
#   elif mode == QualityLevel.MASTER:
#     file_artifact, thumbnail_artifact = output_artifacts
#     write_artifact(file_artifact, image, header)
#     image = resize_array(image, context.master_thumbnail_width)
#     assert not np.isnan(np.min(image))
#     write_artifact(thumbnail_artifact, image, header)
#     return [file_artifact, thumbnail_artifact]

#   raise ValueError(f'Uknown quality level {mode}')
