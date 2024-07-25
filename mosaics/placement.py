import ray
import os
import numpy as np
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator
import largestinteriorrectangle
import cv2 as cv
import psutil
import time
import gc

from plate_solving import solve_field
from decorators import saves_artifacts
from artifact import Artifact
from context import context
from .viewbox import compute_mosaic_viewbox_artifacts
from artifact import read_artifact_image, read_artifact_header, write_artifact, CacheArtifact, FileArtifact

_MEMORY_REQUIREMENT = 25769803776


# TODO: This needs to include the hash of the viewbox.
def mosaic_by_coordinates_artifacts(mode, mosaic_id, filter, stack_id,
                                    input_artifact):
  if mode == 'subframe':
    key = ('placed_image', mosaic_id, stack_id)
    return [CacheArtifact(key, input_artifact.content_hash)]
  elif mode == 'master':
    mosaic = context.mosaics[mosaic_id]
    filename = os.path.join(context.output_dir, 'Mosaics', mosaic.name,
                            f'placed_{mosaic_id}_{filter}_{stack_id}.fit')
    return [FileArtifact(filename, input_artifact.content_hash)]


def _compute_ras_decs(mode, viewbox, multiple=4):
  if mode == 'subframe':
    num_samples_x = context.subframe_mosaic_samples
  elif mode == 'master':
    num_samples_x = context.master_mosaic_samples

  def round_to_nearest_multiple(number):
    return int(np.ceil(number / multiple) * multiple)

  num_samples_x = round_to_nearest_multiple(num_samples_x)
  resolution = viewbox.x_span / num_samples_x
  num_samples_y = round_to_nearest_multiple(viewbox.y_span / resolution)

  xx, yy = np.meshgrid(
      np.linspace(viewbox.x_min, viewbox.x_min + viewbox.x_span, num_samples_x),
      np.linspace(viewbox.y_min, viewbox.y_min + viewbox.y_span, num_samples_y))
  ras, decs = viewbox.projection.unproject(xx, yy)
  ras, decs = np.degrees(ras), np.degrees(decs)

  return ras, decs


def _reshape_tiles(image, tile_size=(32, 32)):
  image_height, image_width = image.shape
  tile_height, tile_width = tile_size
  tiles = image.reshape(image_height // tile_height, tile_height,
                        image_width // tile_width, tile_width)
  return tiles.swapaxes(1, 2)


# Function pseudocode:
#
# Creates the output image as defined by the given grid of RAs and Decs.
#
# For every pixel in the output, grabs the bilinear interpolation of pixels in
# the input image.
#
# Find the largest interior rectangle, and trims all edges to be axis-aligned.
#
# Writes a FITS file to disk that contains the image and some metadata required
# by PhotometricMergeMosaic in PixInsight.
#
# TODO: Provide some feedback on progress.
# TODO: Store these on disk and read them back when unchanged.
@ray.remote(scheduling_strategy="SPREAD")
@saves_artifacts(mosaic_by_coordinates_artifacts)
def mosaic_by_coordinates(mode, mosaic_id, filter, stack_id, input_artifact):
  print(f'### Entering mosaic_by_coordinates '
        f' {mode} {mosaic_id} {filter} {stack_id} {input_artifact}')

  # Wait a random amount of time, then pause until there's at least 16 GB
  # of memory available.
  time.sleep(np.random.randint(1, 5))
  while psutil.virtual_memory().available < _MEMORY_REQUIREMENT:
    print('### placement is waiting until more memory is available')
    time.sleep(5)

  viewbox_key = next(iter(compute_mosaic_viewbox_artifacts(mode,
                                                           mosaic_id))).key
  viewbox = context.get(viewbox_key)
  assert viewbox

  # TODO: It's a bit weird to do plate-solving here. This should all be in a
  # separate function outside of this file.
  #
  # TODO: The subframes need to be calibrated before being placed.
  filename = input_artifact.filename
  image = read_artifact_image(input_artifact)
  header = read_artifact_header(input_artifact)
  plate_solution = solve_field(filename)

  assert image is not None
  assert plate_solution is not None

  print(f'### image.nbytes {image.nbytes / 1024 / 1024 / 1024} GB')

  if plate_solution is None:
    print(f'Could not place artifact {input_artifact} because plate_solution '
          f'was None')
    return

  print(f'mosaic_by_coordinates: image shape {image.shape} dtype {image.dtype} '
        f'min {np.min(image)} max {np.max(image)}')

  wcs = WCS(plate_solution.header)

  # TODO: This seems to take up a huge amount of memory with method 'cubic.'
  # Maybe tiling would reduce the intermediate memory usage.
  source_height, source_width = image.shape
  interp = RegularGridInterpolator((range(source_height), range(source_width)),
                                   image,
                                   method='linear',
                                   bounds_error=False,
                                   fill_value=np.nan)
  del image
  gc.collect()

  multiple = 4

  ras, decs = _compute_ras_decs(mode, viewbox, multiple)
  projected_image = np.zeros_like(ras)
  print(f'### ras.nbytes {ras.nbytes / 1024 / 1024 / 1024} GB')
  print(f'### decs.nbytes {decs.nbytes / 1024 / 1024 / 1024} GB')

  viewbox_height, viewbox_width = ras.shape
  tile_height = int(viewbox_height / multiple)
  tile_width = int(viewbox_width / multiple)
  print(f'### viewbox_height {viewbox_height} tile_height {tile_height}')
  print(f'### viewbox_width {viewbox_width} tile_width {tile_width}')
  ra_tiles = _reshape_tiles(ras, (tile_height, tile_width))
  dec_tiles = _reshape_tiles(decs, (tile_height, tile_width))
  projected_image_tiles = _reshape_tiles(projected_image,
                                         (tile_height, tile_width))

  # TODO: Spread this out with Ray tasks.
  num_tiles_y, num_tiles_x = ra_tiles.shape[0:2]
  for tile_y in range(num_tiles_y):
    for tile_x in range(num_tiles_x):
      print(f'### tile_y {tile_y} tile_x {tile_x}')
      source_xs, source_ys = wcs.all_world2pix(ra_tiles[tile_y, tile_x, :, :],
                                               dec_tiles[tile_y, tile_x, :, :],
                                               1,
                                               quiet=True)
      projected_image_tiles[tile_y, tile_x, :, :] = interp(
          (source_ys, source_xs))

  grid = projected_image != np.nan
  cv_grid = grid.astype("uint8") * 255
  contours, _ = \
      cv.findContours(cv_grid, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
  contour = contours[0][:, 0, :]

  trim_top, trim_left, trim_height, trim_width = largestinteriorrectangle.lir(
      grid, contour)
  print(f'Trimmed trim_top {trim_top} trim_left {trim_left} trim_height '
        f'{trim_height} trim_width {trim_width}')

  trim_bottom = trim_top + trim_height
  trim_right = trim_left + trim_width

  projected_image = np.nan_to_num(projected_image)

  trimmed_image = np.zeros(projected_image.shape, np.float32)
  trimmed_image[trim_left:trim_right,
                trim_top:trim_bottom] = projected_image[trim_left:trim_right,
                                                        trim_top:trim_bottom]
  del projected_image
  gc.collect()

  output_image = trimmed_image
  print(f'### mosaic_by_coordinates: trimming '
        f'output_image dtype {output_image.dtype} '
        f'min {np.min(output_image)} max {np.max(output_image)}')

  output_artifact = next(
      iter(
          mosaic_by_coordinates_artifacts(mode, mosaic_id, filter, stack_id,
                                          input_artifact)))

  if mode == 'subframe':
    write_artifact(output_artifact, output_image)
  elif mode == 'master':
    output_header = {
        'IMAGETYP': 'PlacedMosaic Light',
        'FOCALLEN': float(header['FOCALLEN']),
        'XPIXSZ': float(header['XPIXSZ']),
        'YPIXSZ': float(header['YPIXSZ']),
        'HISTORY': f"TrimMosaicTile.target: foo",
        'HISTORY': f"TrimMosaicTile.top: {trim_top}",
        'HISTORY': f"TrimMosaicTile.bottom: {trim_bottom}",
        'HISTORY': f"TrimMosaicTile.left: {trim_left}",
        'HISTORY': f"TrimMosaicTile.right: {trim_right}"
    }
    write_artifact(output_artifact, output_image, output_header)

    # hdu = pyfits.PrimaryHDU(output_image)
    # hdu.header['SIMPLE'] = True
    # hdu.header['IMAGETYP'] = 'PlacedMosaic Light'
    # hdu.header['FOCALLEN'] = float(header['FOCALLEN'])
    # hdu.header['XPIXSZ'] = float(header['XPIXSZ'])
    # hdu.header['YPIXSZ'] = float(header['YPIXSZ'])
    # hdu.header['HISTORY'] = f"TrimMosaicTile.target: foo"
    # hdu.header['HISTORY'] = f"TrimMosaicTile.top: {trim_top}"
    # hdu.header['HISTORY'] = f"TrimMosaicTile.bottom: {trim_top + trim_height}"
    # hdu.header['HISTORY'] = f"TrimMosaicTile.left: {trim_left}"
    # hdu.header['HISTORY'] = f"TrimMosaicTile.right: {trim_left + trim_width}"

    # # TODO: Is this placed_filename actually checked anywhere?
    # hdu.writeto(output_artifact.filename, overwrite=True)
    # print(f'### Wrote placed image filename {output_artifact.filename}')
