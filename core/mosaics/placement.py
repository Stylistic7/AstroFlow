import ray
import os
import numpy as np
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator
import largestinteriorrectangle
import cv2 as cv
from itertools import product
from pathlib import Path

from plate_solving import solve_field
from artifacts.decorators import saves_artifacts
from artifacts import Artifact
from context import context
from .viewbox import compute_mosaic_viewbox_artifacts
from artifacts import read_artifact_image, read_artifact_header, write_artifact, CacheArtifact, FileArtifact
from hashing import hash_list

_MEMORY_REQUIREMENT = 8 * 1024 * 1024 * 1024  # 8 GB


def mosaic_by_coordinates_artifacts(mode, mosaic_id, filter, stack_id,
                                    input_artifact):
  viewbox_artifact = next(
      iter(compute_mosaic_viewbox_artifacts(mode, mosaic_id)))
  content_hash = hash_list(
      [input_artifact.content_hash, viewbox_artifact.content_hash])

  if mode == 'subframe':
    key = ('placed_image', mosaic_id, filter, stack_id)
    return [CacheArtifact(key, content_hash)]
  elif mode == 'master':
    mosaic = context.mosaics[mosaic_id]
    filename = os.path.join(context.output_dir, 'Mosaics', mosaic.name,
                            f'placed_{mosaic_id}_{filter}_{stack_id}.fit')
    return [FileArtifact(filename, content_hash)]


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


@ray.remote(scheduling_strategy="SPREAD")
def place_tile(tile_y, tile_x, wcs, ra_tiles, dec_tiles, interp):
  ra_tile = ra_tiles[tile_y, tile_x, :, :]
  dec_tile = dec_tiles[tile_y, tile_x, :, :]

  source_xs, source_ys = wcs.all_world2pix(ra_tile, dec_tile, 1, quiet=True)
  return tile_y, tile_x, interp((source_ys, source_xs))


# TODO: Provide some feedback on progress.
# TODO: Store these on disk and read them back when unchanged.
# TODO: Something here is still taking up a huge amount of memory in the Ray
# object store.
# TODO: Placement is perhaps fast enough to not bother storing placed images
# on disk, where they take up a huge amount of space.
@ray.remote(scheduling_strategy="SPREAD", memory=_MEMORY_REQUIREMENT)
@saves_artifacts(mosaic_by_coordinates_artifacts)
def mosaic_by_coordinates(mode, mosaic_id, filter, stack_id, input_artifact):
  print(f'### Entering mosaic_by_coordinates '
        f' {mode} {mosaic_id} {filter} {stack_id} {input_artifact}')

  # TODO: All this Artifact-handling stuff can be streamlined.
  viewbox_artifact = next(
      iter(compute_mosaic_viewbox_artifacts(mode, mosaic_id)))
  viewbox = context.get(viewbox_artifact.key)
  assert viewbox

  # TODO: The subframes need to be calibrated before being placed.
  filename = input_artifact.filename
  image = read_artifact_image(input_artifact)
  header = read_artifact_header(input_artifact)
  metadata = context.metadata
  plate_solution = metadata[input_artifact.content_hash].plate_solution

  assert image is not None
  assert plate_solution is not None

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
                                   fill_value=0)
  del image

  multiple = 4

  ras, decs = _compute_ras_decs(mode, viewbox, multiple)
  projected_image = np.zeros_like(ras)

  viewbox_height, viewbox_width = ras.shape
  tile_height = int(viewbox_height / multiple)
  tile_width = int(viewbox_width / multiple)
  ra_tiles = _reshape_tiles(ras, (tile_height, tile_width))
  dec_tiles = _reshape_tiles(decs, (tile_height, tile_width))
  projected_image_tiles = _reshape_tiles(projected_image,
                                         (tile_height, tile_width))

  num_tiles_y, num_tiles_x = ra_tiles.shape[0:2]
  wcs_ref = ray.put(wcs)
  ra_tiles_ref = ray.put(ra_tiles)
  dec_tiles_ref = ray.put(dec_tiles)
  interp_ref = ray.put(interp)
  remaining_refs = [
      place_tile.remote(tile_y, tile_x, wcs_ref, ra_tiles_ref, dec_tiles_ref,
                        interp_ref)
      for tile_y, tile_x in product(range(num_tiles_y), range(num_tiles_x))
  ]
  while remaining_refs:
    ready_refs, remaining_refs = ray.wait(remaining_refs, num_returns=1)
    tile_y, tile_x, value = ray.get(ready_refs[0])
    projected_image_tiles[tile_y, tile_x, :, :] = value

  print('### projected_image shape', projected_image.shape)

  output_artifacts = mosaic_by_coordinates_artifacts(mode, mosaic_id, filter,
                                                     stack_id, input_artifact)
  output_artifact = next(iter(output_artifacts))

  if mode == 'subframe':
    output_image = projected_image
    write_artifact(output_artifact, output_image)
  elif mode == 'master':
    grid = projected_image != 0
    cv_grid = grid.astype("uint8") * 255
    contours, _ = cv.findContours(cv_grid, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contour = contours[0][:, 0, :]

    trim_left, trim_top, trim_width, trim_height = largestinteriorrectangle.lir(
        grid, contour)
    print(f'Trimmed trim_top {trim_top} trim_left {trim_left} trim_height '
          f'{trim_height} trim_width {trim_width}')

    trim_bottom = trim_top + trim_height
    trim_right = trim_left + trim_width

    if trim_left == 0 and trim_top == 0 and trim_width == projected_image.shape[
        1] and trim_height == projected_image.shape[0]:
      assert False, "largestinteriorrectangle matched entire image"

    trimmed_image = np.zeros(projected_image.shape, np.float32)
    trimmed_image[trim_top:trim_bottom,
                  trim_left:trim_right] = projected_image[trim_top:trim_bottom,
                                                          trim_left:trim_right]
    del projected_image
    output_image = trimmed_image

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

    print(f'### placement done; output_artifact is {output_artifact}')

  return output_artifacts

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
