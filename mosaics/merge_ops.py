import os
import numpy as np
import ray

from .classes import MergeOp, Panel
from .merge_images import merge_images
from artifact import Artifact
from context import context
from hashing import hash_list
from file_io.generic import get_content_hash
from decorators import saves_artifacts
from dependency_graph import Graph, print_graph


def fill_template(template, value):
  return tuple(value if t is None else t for t in template)


# TODO: pairs is a misnomer, because the end of an odd-length axis is a single
# panel.
def make_pairs(min_index, max_index, template):
  pairs = [[fill_template(template, i),
            fill_template(template, i + 1)]
           for i in list(range(min_index, max_index, 2))]

  num = max_index - min_index + 1
  if num % 2 == 1:
    pairs.append([fill_template(template, max_index)])

  return pairs


# TODO: Strongly-type this. Panels is a dictionary from tile indices to
# Panel objects.
def make_merge_ops(mode, mosaic_id, filter, panels, from_task):
  graph = Graph()
  merge_ops = []

  def make_pair_id(level, pair):

    def join_tile_indices(tile_indices):
      return '_'.join([str(i) for i in tile_indices])

    return str(level) + '__' + '__'.join([join_tile_indices(p) for p in pair])

  def process_pairs(level, pairs, template, new_panels):
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
        content_hash = hash_list(panels[p].content_hash for p in pair)

        # TODO: This goes outside the artifacts workflow.
        image_key = ('merged_images', mode, mosaic_id, filter, pair_id,
                     content_hash)

        output = Panel(mosaic_id=mosaic_id,
                       filter=filter,
                       merge_op_id=pair_id,
                       content_hash=content_hash,
                       image_key=image_key)
        merge_op = MergeOp(input_artifacts=[input_0, input_1],
                           output_artifact=output)
        merge_task = (merge_images, mode, mosaic_id, input_0, input_1, output)
        from_task[output] = merge_task
        new_panels[tile_indices] = output
        print(f'### making merge op for pair {pair}')
        print(f'### input_0 {input_0}')
        print(f'### input_1 {input_1}')
        merge_ops.append(merge_op)
        dependencies = set()
        if input_0 in from_task:
          dependencies.add(from_task[input_0])
        else:
          print(f'### panel {input_0} was not in from_task')
        if input_1 in from_task:
          dependencies.add(from_task[input_1])
        else:
          print(f'### panel {input_1} was not in from_task')
        graph[merge_task] = dependencies

  level = 0
  while len(panels) > 1:
    xs = [t[0] for t in panels.keys()]
    ys = [t[1] for t in panels.keys()]
    min_x_index, max_x_index = np.min(xs), np.max(xs)
    min_y_index, max_y_index = np.min(ys), np.max(ys)

    new_panels = {}
    if (max_x_index - min_x_index) > (max_y_index - min_y_index):
      for y in range(min_y_index, max_y_index + 1):
        template = (None, y)
        pairs = make_pairs(min_x_index, max_x_index, template)
        process_pairs(level, pairs, template, new_panels)
    else:
      for x in range(min_x_index, max_x_index + 1):
        template = (x, None)
        pairs = make_pairs(min_y_index, max_y_index, template)
        process_pairs(level, pairs, template, new_panels)

    panels = new_panels
    level += 1

  return merge_ops, graph
