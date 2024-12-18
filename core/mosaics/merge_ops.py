import os
import numpy as np
import ray
from typing import Any, TypeAlias

from context import context
from dependency_graph import Graph, TaskDescriptor
from .classes import MergeOp, Panel
from .merge_images import merge_images
from artifacts import CacheArtifact, FileArtifact, Artifact
from hashing import hash_list
from previews import QualityLevel
from mosaics.types import MosaicHash, Template, TileIndices, Pair
from mosaics.mosaic import Mosaic
from dependency_graph.astroflow_node import AstroFlowNode


# TODO: Can these types be made more specific?
def fill_template(template: Template, value: int) -> TileIndices:
  x, y = template
  result: TileIndices = (x if x is not None else value,
                         y if y is not None else value)
  return result


# TODO: pairs is a misnomer, because the end of an odd-length axis is a single
# panel.
def make_pairs(min_index: int, max_index: int,
               template: Template) -> list[Pair]:
  pairs = [[fill_template(template, i),
            fill_template(template, i + 1)]
           for i in list(range(min_index, max_index, 2))]

  num = max_index - min_index + 1
  if num % 2 == 1:
    pairs.append([fill_template(template, max_index)])

  return pairs


# Ultimately, all the matters here is the output Graph. In the new dependency
# graph structure, it could be something like this:
#
# def make_merge_graph(mode, mosaic_id, filter, tile_indices, *images):
#   ...
#
# Where the graph is constructed something like this:
#
# Graph({(make_merge_graph, mode, mosaic_id, filter, mosaic.tile_indices): [
#   (post_process_stack, stack_id) for stack_id in mosaic.stack_ids
# ]})


@AstroFlowNode()
def make_merge_ops(mode: QualityLevel, mosaic: Mosaic, filter: str,
                   *placed_images: list[Any]) -> Graph:
  graph = Graph()
  merge_ops = []

  assert len(mosaic.tile_indices) == len(placed_images)
  # panels = [
  #     Panel(tile_indices=t, image=i)
  #     for t, i in zip(mosaic.tile_indices, placed_images)
  # ]
  panels = {
      t: Panel(tile_indices=t, image=i)
      for t, i in zip(mosaic.tile_indices, placed_images)
  }

  def make_pair_id(level: int, pair: Pair) -> str:

    def join_tile_indices(tile_indices: TileIndices) -> str:
      return '_'.join([str(i) for i in tile_indices])

    return str(level) + '__' + '__'.join([join_tile_indices(p) for p in pair])

  def process_pairs(level: int, pairs: list[Any], template: Template,
                    new_panels: dict[TileIndices, Panel]) -> None:
    for i, pair_in in enumerate(pairs):
      pair = []
      for p in pair_in:
        if p in panels:
          pair.append(p)

      tile_indices = fill_template(template, i)
      if len(pair) == 0:
        continue
      elif len(pair) == 1:
        new_panels[tile_indices] = panels[pair[0]]
      elif len(pair) == 2:
        pair_id = make_pair_id(level, pair)
        input_0 = panels[pair[0]]
        input_1 = panels[pair[1]]
        print(f'### process_pairs input_0 {input_0}')
        print(f'### process_pairs input_1 {input_1}')
        # content_hashes = [panels[p].content_hash for p in pair]
        # print(f'### content_hashes {content_hashes}')
        # content_hash = hash_list(content_hashes)

        output_image = "foo.fits"

        # if mode == QualityLevel.SUBFRAME:
        #   output_image = CacheArtifact(
        #       ('merged_images', mode, mosaic.stable_hash, filter, pair_id),
        #       mosaic.content_hash)
        # elif mode == QualityLevel.MASTER:
        #   output_image = os.path.join(
        #       context.output_dir, 'Mosaics', mosaic.name,
        #       f'merged_{mosaic.stable_hash}_{filter}_{pair_id}.fit')

        # TODO: Is there any reason to have both the MergeOps and also the
        # merge_images tasks in a Graph?
        output = Panel(tile_indices=tile_indices, image=output_image)
        merge_op = MergeOp(input_images=[input_0.image, input_1.image],
                           output_image=output_image)
        merge_ops.append(merge_op)
        merge_task = (merge_images, mode, mosaic.stable_hash, input_0.image,
                      input_1.image, output_image)
        new_panels[tile_indices] = output
        print(f'### making merge op for pair {pair}')
        print(f'### input_0 {input_0}')
        print(f'### input_1 {input_1}')
        # dependencies = set()
        # if input_0 in from_task:
        #   dependencies.add(from_task[input_0])
        # else:
        #   print(f'### panel {input_0} was not in from_task')
        # if input_1 in from_task:
        #   dependencies.add(from_task[input_1])
        # else:
        #   print(f'### panel {input_1} was not in from_task')
        # graph[merge_task] = dependencies

  level = 0
  while len(panels) > 1:
    xs = [t[0] for t in panels.keys()]
    ys = [t[1] for t in panels.keys()]
    min_x_index, max_x_index = np.min(xs), np.max(xs)
    min_y_index, max_y_index = np.min(ys), np.max(ys)

    new_panels: dict[TileIndices, Panel] = {}
    if (max_x_index - min_x_index) > (max_y_index - min_y_index):
      for y in range(min_y_index, max_y_index + 1):
        template: Template = (None, y)
        pairs = make_pairs(min_x_index, max_x_index, template)
        print('### pairs', pairs)
        process_pairs(level, pairs, template, new_panels)
    else:
      for x in range(min_x_index, max_x_index + 1):
        template = (x, None)
        pairs = make_pairs(min_y_index, max_y_index, template)
        print('### pairs', pairs)
        process_pairs(level, pairs, template, new_panels)

    panels = new_panels
    level += 1

  return merge_ops, graph
