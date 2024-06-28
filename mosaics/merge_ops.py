import os
import numpy as np
import ray

import pix
import hashing
from . import classes

from artifact import Artifact
from context import context
from hashing import hash_list
from file_io.generic import get_content_hash
from decorators import saves_artifacts
from dependency_graph import print_graph
import astropy.io.fits as pyfits  # type: ignore


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


def _should_run(node):
  task, *args = node
  print(f'### _should_run in merge_ops: task {task} args {args}')
  return task.should_run(*args)


# TODO: Strongly-type this. Panels is a dictionary from tile indices to
# Artifacts, which are dictionaries of filenames, content hashes, and creator
# tasks.
def new_make_merge_ops(mosaic_id, panels, from_task):
  graph = {}
  merge_ops = []

  def make_pair_id(level, pair):

    def join_tile_indices(tile_indices):
      return '_'.join([str(i) for i in tile_indices])

    return str(level) + '__' + '__'.join([join_tile_indices(p) for p in pair])

  def process_pairs(level, pairs, template, new_panels):
    for i, pair in enumerate(pairs):
      tile_indices = fill_template(template, i)
      if len(pair) == 1:
        new_panels[tile_indices] = panels[pair[0]]
      elif len(pair) == 2:
        pair_id = make_pair_id(level, pair)
        input_0 = panels[pair[0]]
        input_1 = panels[pair[1]]
        content_hash = hash_list(panels[p].content_hash for p in pair)
        output = Artifact(filename=f'merged_{mosaic_id}_{pair_id}.fit',
                          content_hash=content_hash)
        # TODO: Maybe use 'id' instead of 'filename' in the artifacts, since
        # some may be stored in the cache instead of directly on disk.
        merge_op = classes.MergeOp(input_artifacts=[input_0, input_1],
                                   output_artifact=output)
        merge_task = (merge_images, 'master', mosaic_id, input_0, input_1,
                      output)
        from_task[output] = merge_task
        new_panels[tile_indices] = output
        print(f'### making merge op for pair {pair}')
        print(f'### panels[pair[0]] {panels[pair[0]]}')
        print(f'### panels[pair[1]] {panels[pair[1]]}')
        merge_ops.append(merge_op)
        if merge_images.should_run('master', mosaic_id, input_0, input_1,
                                   output):
          dependencies = []
          if _should_run(from_task[input_0]):
            dependencies.append(from_task[input_0])
          if _should_run(from_task[input_1]):
            dependencies.append(from_task[input_1])
          if dependencies:
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


# TODO: Deprecate.
# def make_merge_ops(mode, mosaic_id, panels, output_dir):
#   print(f'Panels are {panels}')

#   filters = set(p.filter for p in panels)
#   assert len(filters) == 1
#   filter = list(filters)[0]

#   def get_panel(tile_indices):
#     return next(p for p in panels if list(p.tile_indices) == list(tile_indices))

#   def make_pair_id(pair):
#     panels = [get_panel(p) for p in pair]

#     def join_tile_indices(tile_indices):
#       return '_'.join([str(i) for i in tile_indices])

#     return '__'.join([join_tile_indices(p.tile_indices) for p in panels])

#   # TODO: Do we need to cache merged sub-results? Merging is so fast that
#   # maybe it's not necessary.
#   merge_ops = []
#   level = 0
#   while len(panels) > 1:
#     min_x_index = np.min([p.tile_indices[0] for p in panels])
#     max_x_index = np.max([p.tile_indices[0] for p in panels])
#     min_y_index = np.min([p.tile_indices[1] for p in panels])
#     max_y_index = np.max([p.tile_indices[1] for p in panels])
#     print(f'Mosaic x axis indices {min_x_index} to {max_x_index}')
#     print(f'Mosaic y axis indices {min_y_index} to {max_y_index}')

#     print(f'Merging panels at level {level}')
#     level += 1

#     def process_pairs(pairs, template):
#       for i, pair in enumerate(pairs):
#         tile_indices = fill_template(template, i)
#         if len(pair) == 1:
#           new_panels.append(
#               classes.Panel(image_filename=get_panel(pair[0]).placed_filename,
#                             content_hash=get_panel(pair[0]).content_hash,
#                             tile_indices=tile_indices))
#         elif len(pair) == 2:
#           # TODO: This needs to be a unique filename, like
#           # merged_master_560c5551fddcef56_B_0_0__0_1_a98afca08fa78b11.fit
#           # I am missing the mosaic id and the content hashes of the placed
#           # images.
#           content_hash = hashing.hash_list(
#               [get_panel(p).content_hash for p in pair])
#           new_image_id = os.path.join(
#               output_dir, f'merged_{mode}_{mosaic_id}_{filter}_'
#               f'{make_pair_id(pair)}_{content_hash}.fit')
#           new_panels.append(
#               classes.Panel(placed_filename=new_image_id,
#                             content_hash=content_hash,
#                             tile_indices=tile_indices))
#           merge_ops.append(
#               classes.MergeOp(
#                   output_filename=new_image_id,
#                   image_filenames=[get_panel(p).placed_filename for p in pair]))

#     new_panels = []
#     if (max_x_index - min_x_index) > (max_y_index - min_y_index):
#       for y in range(min_y_index, max_y_index + 1):
#         template = (None, y)
#         pairs = make_pairs(min_x_index, max_x_index, template)
#         process_pairs(pairs, template)
#     else:
#       for x in range(min_x_index, max_x_index + 1):
#         template = (x, None)
#         pairs = make_pairs(min_y_index, max_y_index, template)
#         process_pairs(pairs, template)

#     panels = new_panels
#     print(f'Panels are now length {len(panels)}')
#     for panel in panels:
#       print(f'{panel}')

#   for merge_op in merge_ops:
#     print(f'merge_op {merge_op}')

#   return merge_ops


@pix.PixInsight()
def run_photometric_mosaic(images, output_filename):
  return pix.PhotometricMosaic(images, output_filename)


def merge_images_artifacts(mode, mosaic_id, input_0, input_1, output):
  print('### merge_images_artifacts, output', output)
  return [output]


# TODO: This is fake. Implement.
@saves_artifacts(merge_images_artifacts)
@ray.remote
def merge_images(mode, mosaic_id, input_0, input_1, output):
  filename = os.path.join(context.output_dir, output.filename)
  image = np.random.random((10, 10))
  hdu = pyfits.PrimaryHDU(image)
  hdu.header['CONTHASH'] = output.content_hash
  hdu.writeto(filename, overwrite=True)
