import io
import os
import objsize
import tempfile
import shutil
import psutil
import subprocess
import time
from pathlib import Path
from xisf import XISF
import plate_solving
import config
import astropy.io.fits as pyfits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import numpy as np
import matplotlib.pyplot as plt
from shapely import Polygon, Point
import traceback
import itertools
import uuid
from enum import Enum
from dataclasses import dataclass
import largestinteriorrectangle
from multiprocessing import Pool
import dependency_graph as dg
from output_file import OutputFile
from prev_results import get_prev_result_object

import pix
import file_io
from mosaic import Mosaic
from cache import cache
import spatial_association
import position_string
import image_handling

from mosaics import classes
from mosaics import dimensions
from mosaics import input_selection
from mosaics import merge_ops
from mosaics import identification
from mosaics import projection
from mosaics import ransac
from mosaics import trim_mosaic_tile

# TODO: Clean up and improve this code.
#
# Split into multiple files:
#
# - One file for identifying overlapping targets
# - One file for finding regular-grid mosaics
# - One file for image selection
# - One file for projecting and trimming panels
# - One file for making pairs and merge ops
# - One top-level entrypoint file

# def _make_integral(L):
#   L -= np.min(L)

#   def mae(L):
#     return np.mean(np.abs(L - np.round(L)))

#   divisors = list(range(1, int(np.ceil(np.max(L))) + 1))

#   if not divisors:
#     return [], np.inf

#   maes = [mae(L / d) for d in divisors]

#   index = np.argmin(maes)
#   integral = np.round(L / divisors[index]).astype(np.int32)

#   return integral, maes[index]


def _error(center, basis_vectors):
  center_in_basis = np.dot(center, basis_vectors)
  return np.mean(np.abs(center_in_basis - np.round(center_in_basis)), axis=0)


def _plot_mosaic(mosaic, plot_title=None):
  plt.figure()
  plt.title(plot_title)

  for target in mosaic:
    plt.plot(*target.footprint_polygon().exterior.xy, color='black')
    plt.scatter([target.center[0]], [target.center[1]], color='black')

  plt.show()


def show_image(data):
  fig = plt.figure(figsize=(10, 10))
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  ax.imshow(data, cmap="gray")
  plt.show()


def process_mosaic_per_filter(mosaic, targets, stacks, filter, output_dir):
  panels = []
  for target_id in mosaic.target_ids:
    target = targets[target_id]

    assert target.tile_indices, f'target {target} has bad tile_indices'

    # TODO: Remove this filter and process all color channels.
    stack = next((s for s in stacks.values()
                  if s.filter == filter and s.id in target.stack_ids), None)

    if stack is None:
      print(f'### Could not find stack')
      return None

    print(f'### Found stack {stack}')

    # prev_result = get_prev_result_object('post_process_stack', stack.id)
    # if prev_result is None:
    #   print(f'### Could not get prev result for stack {stack.id}')
    #   return None
    # filename = prev_result['output_filename']

    # TODO: Add universal mechanism for getting the current highest-quality
    # preview of a stack.
    #
    # Ideas for this:
    #
    # - Simply update ('stack_preview_image', foo) whenever a better-quality
    #   version of the image is generated organically. The final stacked master
    #   will organically replace the autostretched single frame. Downside:
    #   preview is not part of the dependency graph, so I can't automatically run
    #   downstream tasks that depend on the preview.
    #
    # - Make a preview dependency graph that is sort of a "peer" to the main
    #   full-resolution graph. The full-resolution graph has stacking in it,
    #   while the preview graph just has single autostretch images, etc. The
    #   graph can dictate parameters, like output resolution, for the mosaic
    #   stitching.
    obj = get_prev_result_object('stack_subframes', stack.id)
    if not obj or not obj.filename:
      print(f'### Could not get stack master for {stack}')
      return None

    print(f'### Got {obj.filename} for {stack} with id {stack.id}')
    panels.append(
        classes.Panel(image_filename=obj.filename,
                      tile_indices=target.tile_indices))

  print(f'### Mosaic contains {len(panels)} panels with filenames:')
  for p in panels:
    print(f'###   {p.image_filename}')

  with Pool() as p:
    plate_solutions = p.map(plate_solving.solve_field,
                            (p.image_filename for p in panels))

  print(f'### Got {len(plate_solutions)} plate solutions')

  mosaic_dimensions = dimensions.get_mosaic_dimensions(panels, plate_solutions)

  print(f'### Running mosaic_by_coordinates in parallel...')
  with Pool() as p:
    panels = p.starmap(projection.mosaic_by_coordinates,
                       ((p, s, mosaic_dimensions, output_dir)
                        for p, s in zip(panels, plate_solutions)))
  print(f'### mosaic_by_coordinates is done')

  output_image = merge_mosaic(panels, output_dir)

  merged_mosaic = os.path.join(output_dir, f"merged_mosaic_{filter}.fits")
  print(f'Copying final mosaic image to {merged_mosaic}')
  shutil.copy(output_image[0], merged_mosaic)

  return OutputFile(merged_mosaic)


def process_mosaic(mosaic, targets, stacks, output_dir):
  print(f'### Entering process_mosaic for mosaic {mosaic}')

  proc = psutil.Process()
  print(f'### There are {len(proc.open_files())} open files')

  if len(mosaic.target_ids) < 2:
    print(f'No processing to do for mosaic with only '
          f'{len(mosaic.target_ids)} target ids')
    return (dg.Ran.DID_NOT_RUN, (dg.Result.NEVER_RUN, dict()))

  output_dir = os.path.join(output_dir, 'Mosaics', mosaic.name)
  os.makedirs(output_dir, exist_ok=True)

  # TODO: Don't hardcode these filters. Find the set from the images
  # themselves.
  #
  # TODO: Store the filter along with the output filename.
  output_filenames = []
  for filter in ['H', 'S', 'O', 'L', 'R', 'G', 'B']:
    result = process_mosaic_per_filter(mosaic, targets, stacks, filter,
                                       output_dir)
    if result:
      output_filenames.append(result)

  return (dg.Ran.RAN, (dg.Result.SUCCESS,
                       dict(panels=panels, output_filenames=output_filenames)))


def main():
  for mosaic in cache['mosaics']:
    if 'Filamentary' not in mosaic.id or len(mosaic.target_ids) < 2:
      continue

    process_mosaic(mosaic)


if __name__ == '__main__':
  main()