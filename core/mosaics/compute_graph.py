from typing import Any, Optional
from pathlib import Path
import astropy.io.fits as pyfits  # type: ignore
from shapely import box  # type: ignore

from context import context, WriteMode
from customization.types import CustomizedStack
from dependency_graph import Graph, Node, make_graph_from_list
from file_io.conversion import convert_xisf_to_fits
from mosaics.merge_images import merge_images
from mosaics.types import Mosaic, Viewbox
from mosaics.viewbox import compute_mosaic_viewbox
from mosaics.placement import place_image
from mosaics.post_process import post_process_mosaic
from mosaics.trimming import project_to_viewbox, largest_interior_rectangle
from mosaics.types import TileIndices
from stacks.compute_graph import get_stack_image
from stacks.composite_stack import CompositeStacksDict
from stacks.types import StackHash
from previews import QualityLevel
from task.astroflow_node import task
from targets.types import TargetsDict, Target, TargetHash
from file_io.generic import read_image


# TODO: What's supposed to happen with these final artifacts? I guess they're
# just stored in "findable locations" that do not depend on content hashes.
@task.remote
def store_mosaic_color_channel(mode: QualityLevel, mosaic: Mosaic, filter: str,
                               input: Path) -> Optional[Path]:
  if input is None:
    return

  print(f'### store_mosaic_color_channel: filter {filter} input {input}')
  key = ('mosaic_thumbnail_image', mode, mosaic.stable_hash, filter)

  if mode == QualityLevel.SUBFRAME:
    image = read_image(input)
    context.set(key, image, write_mode=WriteMode.DISK_ONLY)
    print(f'### store_mosaic_color_channel wrote key {key}')
    return None
  elif mode == QualityLevel.MASTER:
    context.set(key, input, write_mode=WriteMode.DISK_ONLY)
    filename = Path(
        context.output_dir
    ) / 'Mosaics' / mosaic.name / f'final_merged_{mosaic.stable_hash}_{filter}.fits'
    filename.parent.mkdir(parents=True, exist_ok=True)
    convert_xisf_to_fits(input, filename)
    return [filter, filename]


def targets_for_mosaic(mosaic: Mosaic, targets: TargetsDict) -> TargetsDict:
  '''Returns only the targets included in the mosaic.'''
  return TargetsDict(
      {target_id: targets[target_id] for target_id in mosaic.target_ids})


def stacks_for_mosaic(mosaic: Mosaic, stacks: CompositeStacksDict,
                      targets: TargetsDict) -> CompositeStacksDict:
  '''Returns only the stacks included in the mosaic.'''
  return CompositeStacksDict({
      stack_id: stacks[stack_id] for target_id in mosaic.target_ids
      for stack_id in targets[target_id].stack_ids
  })


@task.remote
def build_mosaic_subgraph(mode: QualityLevel, mosaic: Mosaic,
                          stacks: CompositeStacksDict,
                          targets: TargetsDict) -> Graph:
  print(f'### build_mosaic_subgraph mode {mode} for mosaic {mosaic}')
  viewbox = compute_mosaic_viewbox(mode, mosaic)
  assert targets
  assert stacks

  filters = set(stack.filter for stack in stacks.values())

  mask_task = (compute_mosaic_masks, mode, mosaic, viewbox)

  graph = Graph()
  store_tasks = []
  for filter in filters:
    strips = get_stacks_for_filter(mosaic, filter, targets, stacks)
    print(f'### strips for filter {filter} {strips}')

    if mode == QualityLevel.SUBFRAME:
      # In subframe mode, missing panels are allowed. Just remove them from
      # the strips.
      strips = tuple(
          tuple(s for s in strip if s is not None) for strip in strips)
      strips = tuple(s for s in strips if len(s))
    else:
      # In master mode, missing panels are not allowed; skip the filter.
      if any(None in strip for strip in strips):
        print(f'### Skipping mosaic {mosaic} filter {filter} because '
              f'some stacks are missing')
        continue

    store_task = (store_mosaic_color_channel, mode, mosaic, filter)
    filter_subgraph_task = Node(build_mosaic_filter_subgraph, mode, mosaic,
                                strips, stacks, filter, viewbox)
    graph[store_task] = {filter_subgraph_task}
    graph[filter_subgraph_task] = {mask_task}
    store_tasks.append(store_task)

  graph[(post_process_mosaic, mode, mosaic)] = store_tasks

  return graph


# TODO: This function seems to be slow for some reason.
@task.remote(num_cpus=0.5)
def build_placement_subgraph(mode: QualityLevel, mosaic: Mosaic,
                             stack: CustomizedStack, filter: str,
                             tile_indices: TileIndices, viewbox: Viewbox,
                             mask: box):
  # TODO: Integrate the stack mask here too.
  assert stack is not None
  return make_graph_from_list([(get_stack_image, mode, stack),
                               Node(place_image, mode, mosaic, stack,
                                    tile_indices, filter, viewbox, mask)])


def compute_mask(strip: list[TargetHash], viewbox: Viewbox) -> box:
  targets = context.targets
  print(f'### Entering compute_mask')

  # Compute the 2D polygon of all the targets in the strip.
  strip_poly = None
  for target_id in strip:
    target_poly = project_to_viewbox(targets[target_id].footprint, viewbox)
    strip_poly = target_poly if strip_poly is None else strip_poly.union(
        target_poly)

  print(f'### Calling largest_interior_rectangle')
  return largest_interior_rectangle(strip_poly)


@task.remote
def compute_mosaic_masks(mode: QualityLevel, mosaic: Mosaic,
                         viewbox: Viewbox) -> list[box]:
  if mode == QualityLevel.MASTER:
    return [compute_mask(strip, viewbox) for strip in mosaic.strips]


@task.remote
def compute_target_mask(target: Target, viewbox: Viewbox) -> box:
  return largest_interior_rectangle(
      project_to_viewbox(target.footprint, viewbox))


def get_stacks_for_filter(
    mosaic: Mosaic, filter: str, targets: TargetsDict,
    stacks: CompositeStacksDict) -> Optional[list[list[StackHash]]]:
  '''Returns the mosaics strips with target hashes replaced by stack hashes.'''

  def get_stack_for_target(target_id: TargetHash) -> StackHash:
    for stack_id in targets[target_id].stack_ids:
      if stacks[stack_id].filter == filter:
        return stack_id

  def get_stacks_for_strip(strip: list[TargetHash]) -> list[StackHash]:
    return tuple(get_stack_for_target(target_id) for target_id in strip)

  return tuple(get_stacks_for_strip(strip) for strip in mosaic.strips)


@task.remote
def build_mosaic_filter_subgraph(mode: QualityLevel, mosaic: Mosaic,
                                 strips: list[list[StackHash]],
                                 stacks: CompositeStacksDict, filter: str,
                                 viewbox: Viewbox,
                                 masks: list[box] | None) -> Graph:

  print(f'### build_mosaic_filter_subgraph mode {mode} '
        f'mosaic {mosaic} filter {filter} strips {mosaic.strips}')

  if masks is None:
    masks = [None] * len(strips)

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
  for strip_index, (strip, mask) in enumerate(zip(strips, masks)):
    prev_task = None
    for i, stack_id in enumerate(strip):
      # Missing stacks are acceptable, except in MASTER mode.
      if stack_id is None:
        if mode == QualityLevel.MASTER:
          print(f'### build_mosaic_filter_subgraph returning None because a '
                f'stack for filter {filter} was not found')
          return None
        else:
          continue

      stack = stacks[stack_id]
      if i == 0:
        prev_task = Node(build_placement_subgraph, mode, mosaic, stack, filter,
                         TileIndices((strip_index, i)), viewbox, mask)
      else:
        placement_task = Node(build_placement_subgraph, mode,
                              mosaic, stack, filter,
                              TileIndices((strip_index, i)), viewbox, mask)
        merge_task = Node(merge_images,
                          mode,
                          mosaic,
                          name=f'merge_{mosaic.stable_hash}_{stack_id}')
        if prev_task is not None:
          graph[placement_task] = [prev_task]
        graph[merge_task] = [prev_task, placement_task]
        prev_task = merge_task

    assert prev_task is not None, f'strips are {strips}'
    strip_tasks.append(prev_task)

  for i in range(len(strips)):
    if i == 0:
      prev_task = strip_tasks[0]
    else:
      merge_task = Node(merge_images,
                        mode,
                        mosaic,
                        name=f'merge_{mosaic.stable_hash}_strip_{i}')
      graph[merge_task] = [prev_task, strip_tasks[i]]
      prev_task = merge_task

  return graph
