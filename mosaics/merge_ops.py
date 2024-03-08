import os
import uuid
import numpy as np
from mosaic import Mosaic
import pix
import plate_solving
from . import classes


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


def make_merge_ops(panels, output_dir):
  print(f'Panels are {panels}')

  def get_panel(tile_indices):
    return next(p for p in panels if list(p.tile_indices) == list(tile_indices))

  merge_ops = []
  level = 0
  while len(panels) > 1:
    min_x_index = np.min([p.tile_indices[0] for p in panels])
    max_x_index = np.max([p.tile_indices[0] for p in panels])
    min_y_index = np.min([p.tile_indices[1] for p in panels])
    max_y_index = np.max([p.tile_indices[1] for p in panels])
    print(f'Mosaic x axis indices {min_x_index} to {max_x_index}')
    print(f'Mosaic y axis indices {min_y_index} to {max_y_index}')

    print(f'Merging panels at level {level}')
    level += 1

    def process_pairs(pairs, template):
      for i, pair in enumerate(pairs):
        tile_indices = fill_template(template, i)
        if len(pair) == 1:
          new_panels.append(
              classes.Panel(image_filename=get_panel(pair[0]).image_filename,
                            tile_indices=tile_indices))
        elif len(pair) == 2:
          new_image_id = os.path.join(output_dir, f'merged_{uuid.uuid4()}.fit')
          new_panels.append(
              classes.Panel(image_filename=new_image_id,
                            tile_indices=tile_indices))
          merge_ops.append(
              classes.MergeOp(
                  output_filename=new_image_id,
                  image_filenames=[get_panel(p).image_filename for p in pair]))

    new_panels = []
    if (max_x_index - min_x_index) > (max_y_index - min_y_index):
      for y in range(min_y_index, max_y_index + 1):
        template = (None, y)
        pairs = make_pairs(min_x_index, max_x_index, template)
        process_pairs(pairs, template)
    else:
      for x in range(min_x_index, max_x_index + 1):
        template = (x, None)
        pairs = make_pairs(min_y_index, max_y_index, template)
        process_pairs(pairs, template)

    panels = new_panels
    print(f'Panels are now length {len(panels)}')
    for panel in panels:
      print(f'{panel}')

  for merge_op in merge_ops:
    print(f'merge_op {merge_op}')

  return merge_ops


@pix.PixInsight
def run_photometric_mosaic(merge_op):
  images = pix.OpenImages(pix.FileList(merge_op.image_filenames),
                          ['placed_0', 'placed_1'])
  merged = pix.PhotometricMosaic(images)
  result = pix.SaveImages(merged, [merge_op.output_filename])

  return result


def merge_mosaic(panels, output_dir):
  print(f'### Entering merge_mosaic, output_dir {output_dir}')
  for merge_op in make_merge_ops(panels, output_dir):
    output_image = run_photometric_mosaic(merge_op)
    print(f'### run_photometric_mosaic returned {output_image}')

  return output_image
