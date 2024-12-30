import ray
import os
import numpy as np
from typing import Any
from pathlib import Path
import astropy.io.fits as pyfits  # type: ignore

from artifacts import Artifact, CacheArtifact, FileArtifact, read_artifact_image, write_artifact, artifact_exists
from context import context, WriteMode
from dependency_graph import Graph, Node, TaskDescriptor, print_graph, make_graph_from_list
# TODO: Unify quarantining
import quarantining
from metadata.classes import ImageMetadata, MetadataDict
from mosaics.types import MosaicHash
from mosaics.classes import Panel, Viewbox
from mosaics.merge_images import merge_images
from mosaics.mosaic import Mosaic
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
from file_io.generic import read_image


@AstroFlowNode()
def mosaic_done(*ancestor_results) -> None:
  pass


# TODO: What's supposed to happen with these final artifacts? I guess they're
# just stored in "findable locations" that do not depend on content hashes.
@AstroFlowNode()
def store_mosaic_color_channel(mode: QualityLevel, mosaic: Mosaic, filter: str,
                               input: Any) -> None:
  print(f'### store_mosaic_color_channel: filter {filter} input {type(input)}')
  key = ('mosaic_thumbnail_image', mode, mosaic.stable_hash, filter)

  if mode == QualityLevel.SUBFRAME:
    context.set(key, input, write_mode=WriteMode.DISK_ONLY)
    print(f'### store_mosaic_color_channel wrote key {key}')
  elif mode == QualityLevel.MASTER:
    image = read_image(input)
    context.set(key, input, write_mode=WriteMode.DISK_ONLY)
    filename = Path(
        context.output_dir
    ) / 'Mosaics' / mosaic.name / f'final_merged_{mosaic.stable_hash}_{filter}.fit'
    hdu = pyfits.PrimaryHDU(image)
    filename.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(str(filename), overwrite=True)


@AstroFlowNode()
def build_mosaic_subgraph(mode: QualityLevel, mosaic: Mosaic) -> Graph:
  print(f'### build_mosaic_subgraph mode {mode} for mosaic {mosaic}')
  viewbox = compute_mosaic_viewbox(mosaic)

  stacks = context.stacks
  targets = context.targets

  print(f'### build_mosaic_subgraph {len(stacks)} stacks')
  print(f'### build_mosaic_subgraph {len(targets)} targets')

  filters = set()
  for target_id in mosaic.target_ids:
    for stack_id in targets[target_id].stack_ids:
      filters.add(stacks[stack_id].filter)
  filters = tuple(filters)

  graph = Graph()
  store_tasks = []
  for filter in filters:
    store_task = (store_mosaic_color_channel, mode, mosaic, filter)
    graph[store_task] = [(build_mosaic_filter_subgraph, mode, mosaic, filter,
                          viewbox)]
    store_tasks.append(store_task)

  graph[mosaic_done] = store_tasks

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


@AstroFlowNode(update_header=False)
def get_best_light_frame_for_target(target: Target,
                                    filter: str) -> FileArtifact | None:
  stacks = context.stacks
  metadata = context.metadata

  stacks_for_filter = [
      s for s in target.stack_ids if stacks[s].filter == filter
  ]
  if not stacks_for_filter:
    print(f'### get_best_light_frame_for_target no stacks for filter {filter}')
    return None

  if len(stacks_for_filter) > 1:
    print(f'### WARNING: Multiple stacks for filter {filter} '
          f'for target {target}')
  stack_id = next(iter(stacks_for_filter))

  print(f'### Found stack_id {stack_id} for filter {filter}')

  best_light_frame = get_best_light_frame_for_stack(stacks[stack_id], metadata)
  return Path(best_light_frame.filename)


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
        f'mosaic {mosaic} filter {filter} strips {mosaic.strips}')

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
        prev_task = Node(build_placement_subgraph, mode, mosaic,
                         targets[target_id], filter,
                         TileIndices((strip_index, i)), viewbox)
      else:
        merge_task = Node(merge_images, mode, name=f'merge_{target_id}')
        graph[merge_task] = [
            prev_task,
            Node(build_placement_subgraph, mode, mosaic, targets[target_id],
                 filter, TileIndices((strip_index, i)), viewbox)
        ]
        prev_task = merge_task

    strip_tasks.append(prev_task)

  for i in range(len(mosaic.strips)):
    if i == 0:
      prev_task = strip_tasks[0]
    else:
      merge_task = Node(merge_images, mode, name=f'merge_strip_{i}')
      graph[merge_task] = [prev_task, strip_tasks[i]]
      prev_task = merge_task

  return graph
