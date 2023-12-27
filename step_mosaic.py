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


def _get_preview_panel_for_stack(stack, image_hashes, plate_solutions):
  light_frames_by_stars = sorted(stack.light_frames,
                                 key=lambda f: plate_solutions[f].num_stars,
                                 reverse=True)
  best_light_frame = light_frames_by_stars[0]
  filename = image_hashes[best_light_frame]
  plate_solution = plate_solutions[best_light_frame]

  return classes.Panel(image_filename=filename, plate_solution=plate_solution)


def _get_master_panel_for_stack(stack):
  obj = get_prev_result_object('stack_subframes', stack.id)
  if not obj or not obj.filename:
    print(f'### Could not get stack master for {stack}: {obj}')
    return None

  print(f'### Got {obj.filename} for {stack} with id {stack.id}')

  plate_solution = plate_solving.solve_field(obj.filename)

  return classes.Panel(image_filename=obj.filename,
                       plate_solution=plate_solution)


def _get_panel_for_stack(stack, image_hashes, plate_solutions, preview):
  if preview:
    return _get_preview_panel_for_stack(stack, image_hashes, plate_solutions)
  else:
    return _get_master_panel_for_stack(stack)


def process_mosaic_per_filter(image_hashes, plate_solutions, mosaic, targets,
                              stacks, filter, preview, output_dir):
  if output_dir:
    cm = contextlib.nullcontext(output_dir)
  else:
    cm = tempfile.TemporaryDirectory()

  with cm as output_dir:
    panels = []
    print(f'### process_mosaic_per_filter mosaic name {mosaic.name}')
    for target_id, tile_indices in zip(mosaic.target_ids, mosaic.tile_indices):
      target = targets[target_id]
      stack = next((s for s in stacks.values()
                    if s.filter == filter and s.id in target.stack_ids), None)
      if not stack:
        return None

      panel = _get_panel_for_stack(stack, image_hashes, plate_solutions,
                                   preview)
      panel.tile_indices = tile_indices
      panels.append(panel)

    print(f'### Mosaic contains {len(panels)} panels with filenames:')
    for p in panels:
      print(f'###   {p.image_filename}')

    mosaic_dimensions = dimensions.get_mosaic_dimensions(panels)

    print(f'### Running mosaic_by_coordinates in parallel...')
    with Pool() as p:
      panels = p.starmap(projection.mosaic_by_coordinates,
                         ((p, mosaic_dimensions, output_dir) for p in panels))
    print(f'### mosaic_by_coordinates is done')

    if preview:
      output_image = file_io.read_fits_image(panels[0].image_filename)
      for p in panels[1:]:
        image = file_io.read_fits_image(p.image_filename)
        output_image = np.maximum(output_image, image)

      return image_handling.array_to_png(output_image)
    else:
      # TODO: This returns a different type than the branch above.
      output_image = merge_ops.merge_mosaic(panels, output_dir)

      merged_mosaic = os.path.join(output_dir, f"merged_mosaic_{filter}.fits")
      print(f'Copying final mosaic image to {merged_mosaic}')
      shutil.copy(output_image[0], merged_mosaic)

      return OutputFile(merged_mosaic)


def process_mosaic(image_hashes,
                   plate_solutions,
                   mosaic,
                   targets,
                   stacks,
                   preview=False,
                   output_dir=None):
  print(f'### Entering process_mosaic for mosaic {mosaic} preview {preview}')

  proc = psutil.Process()
  print(f'### There are {len(proc.open_files())} open files')

  if len(mosaic.target_ids) < 2:
    print(f'No processing to do for mosaic with only '
          f'{len(mosaic.target_ids)} target ids')
    return (dg.Ran.DID_NOT_RUN, (dg.Result.NEVER_RUN, list()))

  if output_dir:
    output_dir = os.path.join(output_dir, 'Mosaics', mosaic.name)
    os.makedirs(output_dir, exist_ok=True)

  # TODO: Don't hardcode these filters. Find the set from the images
  # themselves.
  outputs = []
  for filter in ['H', 'S', 'O', 'L', 'R', 'G', 'B']:
    result = process_mosaic_per_filter(image_hashes, plate_solutions, mosaic,
                                       targets, stacks, filter, preview,
                                       output_dir)
    if result:
      outputs.append(dict(filter=filter, result=result))

  return (dg.Ran.RAN, (dg.Result.SUCCESS, outputs))


def main():
  for mosaic in cache['mosaics']:
    if 'Filamentary' not in mosaic.id or len(mosaic.target_ids) < 2:
      continue

    process_mosaic(mosaic)


if __name__ == '__main__':
  main()