from typing import Any
from pathlib import Path
import astropy.io.fits as pyfits  # type: ignore

from context import context, WriteMode
from customization.types import CustomizedStack
from dependency_graph import Graph, Node, make_graph_from_list
from mosaics.merge_images import merge_images
from mosaics.types import Mosaic, Viewbox
from mosaics.viewbox import compute_mosaic_viewbox
from mosaics.placement import place_small_image, place_large_image
from mosaics.trimming import trim_xisf_image, project_to_viewbox
from mosaics.types import TileIndices
from stacks.compute_graph import get_stack_image
from previews import QualityLevel
from dependency_graph.astroflow_node import AstroFlowNode
from targets.types import Target
from file_io.generic import read_image


@AstroFlowNode()
def mosaic_done(mode: QualityLevel, mosaic: Mosaic) -> None:
  print(f'### mosaic_done mode {mode} mosaid {mosaic}')


# TODO: What's supposed to happen with these final artifacts? I guess they're
# just stored in "findable locations" that do not depend on content hashes.
@AstroFlowNode()
def store_mosaic_color_channel(mode: QualityLevel, mosaic: Mosaic, filter: str,
                               input: Path) -> None:
  print(f'### store_mosaic_color_channel: filter {filter} input {input}')
  key = ('mosaic_thumbnail_image', mode, mosaic.stable_hash, filter)

  if mode == QualityLevel.SUBFRAME:
    image = read_image(input)
    context.set(key, image, write_mode=WriteMode.DISK_ONLY)
    print(f'### store_mosaic_color_channel wrote key {key}')
  elif mode == QualityLevel.MASTER:
    image = read_image(input)
    context.set(key, input, write_mode=WriteMode.DISK_ONLY)
    filename = Path(
        context.output_dir
    ) / 'Mosaics' / mosaic.name / f'final_merged_{mosaic.stable_hash}_{filter}.fits'
    hdu = pyfits.PrimaryHDU(image)
    filename.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(str(filename), overwrite=True)


@AstroFlowNode(cache_result=True)
def build_mosaic_subgraph(mode: QualityLevel, mosaic: Mosaic) -> Graph:
  print(f'### build_mosaic_subgraph mode {mode} for mosaic {mosaic}')
  viewbox = compute_mosaic_viewbox(mode, mosaic)

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
    graph[store_task] = [
        Node(build_mosaic_filter_subgraph,
             mode,
             mosaic,
             filter,
             viewbox,
             priority=2)
    ]
    store_tasks.append(store_task)

  graph[(mosaic_done, mode, mosaic)] = store_tasks

  return graph


def _get_stack_for_filter(target: Target, filter: str):
  stacks = context.stacks

  stacks_for_filter = [
      s for s in target.stack_ids if stacks[s].filter == filter
  ]
  if not stacks_for_filter:
    return None

  if len(stacks_for_filter) > 1:
    print(f'### WARNING: Multiple stacks for filter {filter} '
          f'for target {target}')

  stack_id = next(iter(stacks_for_filter))
  return stacks[stack_id]


# TODO: This function seems to be slow for some reason.
@AstroFlowNode(num_cpus=0.5)
def build_placement_subgraph(mode: QualityLevel, mosaic: Mosaic,
                             stack: CustomizedStack, filter: str,
                             tile_indices: TileIndices, viewbox: Viewbox):
  place_image_fn = place_large_image if mode == QualityLevel.MASTER else place_small_image
  return make_graph_from_list([(get_stack_image, mode, stack),
                               Node(place_image_fn,
                                    mode,
                                    mosaic,
                                    stack,
                                    tile_indices,
                                    filter,
                                    viewbox,
                                    priority=3)])


@AstroFlowNode(num_cpus=0.5)
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
    strip_poly = None
    for i, target_id in enumerate(strip):
      # TODO: This should maybe be a hard failure in master mode.
      if not (stack := _get_stack_for_filter(targets[target_id], filter)):
        continue

      if i == 0:
        strip_poly = project_to_viewbox(stack.footprint, viewbox)
        prev_task = Node(build_placement_subgraph,
                         mode,
                         mosaic,
                         stack,
                         filter,
                         TileIndices((strip_index, i)),
                         viewbox,
                         priority=2)
      else:
        strip_poly = strip_poly.union(
            project_to_viewbox(stack.footprint, viewbox))
        merge_task = Node(merge_images,
                          mode,
                          mosaic,
                          name=f'merge_{mosaic.stable_hash}_{target_id}',
                          priority=2)
        trim_task = Node(trim_xisf_image,
                         strip_poly,
                         name=f'trim_merge_{mosaic.stable_hash}_{target_id}')
        graph[merge_task] = [
            prev_task,
            Node(build_placement_subgraph,
                 mode,
                 mosaic,
                 stack,
                 filter,
                 TileIndices((strip_index, i)),
                 viewbox,
                 priority=2)
        ]
        graph[trim_task] = [merge_task]
        prev_task = trim_task
        # prev_task = merge_task

    strip_tasks.append(prev_task)

  for i in range(len(mosaic.strips)):
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
