import ray
import os
import numpy as np
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator
import largestinteriorrectangle
import astropy.io.fits as pyfits
from spherical_geometry.polygon import SphericalPolygon, SingleSphericalPolygon
from shapely.geometry import Polygon, Point
import astropy.io.fits as pyfits
import time
from pathlib import Path

from file_io.generic import read_header
from plate_solving import solve_field
from image_previews import array_to_png, autostretch
from decorators import saves_artifacts
from artifact import Artifact
from context import context
from file_io.generic import read_image
from .viewbox import compute_mosaic_viewbox_artifacts
from image_metadata.actor import get_metadata
from artifact import read_artifact_image


# TODO: This needs to include the hash of the viewbox.
def mosaic_by_coordinates_artifacts(mode, mosaic_id, stack_id, input_artifact):
  if mode == 'subframe':
    key = ('placed_image', mosaic_id, stack_id, input_artifact.content_hash)
    return [Artifact(key=key)]
  elif mode == 'master':
    pass
    # return [
    #     Artifact(filename=panel.placed_filename,
    #              content_hash=panel.content_hash)
    # ]


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
@ray.remote
@saves_artifacts(mosaic_by_coordinates_artifacts)
def mosaic_by_coordinates(mode, mosaic_id, stack_id, input_artifact):
  print(f'### Entering mosaic_by_coordinates '
        f' {mode} {mosaic_id} {stack_id} {input_artifact}')

  # TODO: Pass this through the context.
  # autostretch = False

  viewbox_key = next(iter(compute_mosaic_viewbox_artifacts(mode,
                                                           mosaic_id))).key
  viewbox = context.get(viewbox_key)
  assert viewbox

  start = time.time()

  dest_xs = viewbox['dest_xs']
  dest_ys = viewbox['dest_ys']
  ras = viewbox['ras']
  decs = viewbox['decs']

  projected_image = np.zeros(ras.shape)
  projection_map = np.zeros(ras.shape)

  # TODO: It's a bit weird to do this here. This should all be in a separate
  # function outside of this file.
  #
  # TODO: The subframes need to be calibrated before being placed.
  filename = input_artifact.filename
  image = read_artifact_image(input_artifact)
  plate_solution = solve_field(filename)

  assert image is not None
  assert plate_solution is not None

  # if panel.image_hash:
  #   filename = image_hashes[panel.image_hash]
  #   header = fits_headers[panel.image_hash]
  #   plate_solution = plate_solutions[panel.image_hash]
  # else:
  #   # TODO: It's a bit weird to do this here. This should all be in a separate
  #   # function outside of this file.
  #   filename = panel.input_filename
  #   header = read_header(filename)
  #   plate_solution = solve_field(filename)

  # if header is None:
  #   print(f'Could not place panel {panel} because header was None')
  #   return

  if plate_solution is None:
    print(f'Could not place artifact {input_artifact} because plate_solution '
          f'was None')
    return

  print(f'mosaic_by_coordinates image dtype {image.dtype} '
        f'min {np.min(image)} max {np.max(image)}')

  # if autostretch:
  #   try:
  #     image = image_previews.autostretch(image)
  #   except Exception as e:
  #     print(f'failed to autostretch, exception {e}')

  print(f'Image has shape {image.shape} and dtype {image.dtype}')

  wcs = WCS(plate_solution.header)

  source_height, source_width = image.shape
  interp = RegularGridInterpolator((range(source_height), range(source_width)),
                                   image,
                                   method='linear',
                                   bounds_error=False,
                                   fill_value=-np.inf)

  # Filter out points that aren't in this panel's footprint.
  assert dest_xs.shape == dest_ys.shape == ras.shape == decs.shape
  footprint = SingleSphericalPolygon.from_wcs(
      pyfits.Header(plate_solution.header))
  if not footprint.is_clockwise():
    footprint = SingleSphericalPolygon(footprint.points[::-1])
  assert not SphericalPolygon.self_intersect(footprint.points)
  print(f'dest_xs.shape {dest_xs.shape} dest_ys.shape {dest_ys.shape}')
  print(f'ras.shape {ras.shape} decs.shape {decs.shape}')
  try:
    dest_xs, dest_ys, ras, decs = zip(
        *[(dx, dy, ra, dec)
          for dx, dy, ra, dec in zip(dest_xs.flatten(), dest_ys.flatten(),
                                     ras.flatten(), decs.flatten())
          if footprint.contains_lonlat(ra, dec)])
  except Exception as e:
    print(f'### Failed to filter points with exception {e}')
    return

  source_xs, source_ys = wcs.all_world2pix(ras, decs, 1, quiet=True)

  for dy, dx, sy, sx in zip(dest_ys, dest_xs, source_ys, source_xs):
    value = interp((sy, sx))
    if value == -np.inf:
      continue

    projected_image[dx, dy] = value
    projection_map[dx, dy] = 128

  end = time.time()
  print(f'Finished projecting image {filename} '
        f'time {end-start:.2f}')
  assert np.count_nonzero(projection_map), "projection failed"

  # TODO: This is surprisingly slow, dominating the projection for fine
  # output resolutions.
  start = time.time()
  trim_top, trim_left, trim_height, trim_width = largestinteriorrectangle.lir(
      projection_map > 0)
  print(f'Trimmed trim_top {trim_top} trim_left {trim_left} trim_height '
        f'{trim_height} trim_width {trim_width}')

  projection_map[trim_left:trim_left + trim_width,
                 trim_top:trim_top + trim_height] = 255

  trimmed_image = np.zeros(projected_image.shape, np.float32)
  trimmed_image[trim_left:trim_left + trim_width, trim_top:trim_top +
                trim_height] = projected_image[trim_left:trim_left + trim_width,
                                               trim_top:trim_top + trim_height]

  output_image = trimmed_image
  end = time.time()
  print(f'mosaic_by_coordinates output_image dtype {output_image.dtype} '
        f'min {np.min(output_image)} max {np.max(output_image)} '
        f'time {end-start:.2f}')

  if mode == 'subframe':
    key = next(
        iter(
            mosaic_by_coordinates_artifacts(mode, mosaic_id, stack_id,
                                            input_artifact))).key
    print(f'### mosaic_by_coordinates artifact key {key}')
    assert key is not None
    filename = '/tmp/foo/' + '_'.join(key) + '.png'
    open(filename, 'wb').write(array_to_png(autostretch(output_image)))
    context[key] = output_image
    print(f'### mosaic_by_coordinates wrote key {key} '
          f'of tyoe {type(output_image)}')
  elif mode == 'master':
    hdu = pyfits.PrimaryHDU(output_image)
    hdu.header['SIMPLE'] = True
    hdu.header['IMAGETYP'] = 'PlacedMosaic Light'
    hdu.header['FOCALLEN'] = float(header['FOCALLEN'])
    hdu.header['XPIXSZ'] = float(header['XPIXSZ'])
    hdu.header['YPIXSZ'] = float(header['YPIXSZ'])
    hdu.header['HISTORY'] = f"TrimMosaicTile.target: foo"
    hdu.header['HISTORY'] = f"TrimMosaicTile.top: {trim_top}"
    hdu.header['HISTORY'] = f"TrimMosaicTile.bottom: {trim_top + trim_height}"
    hdu.header['HISTORY'] = f"TrimMosaicTile.left: {trim_left}"
    hdu.header['HISTORY'] = f"TrimMosaicTile.right: {trim_left + trim_width}"

    # TODO: Is this placed_filename actually checked anywhere?
    hdu.writeto(panel.placed_filename, overwrite=True)
    print(f'### Wrote placed image filename {panel.placed_filename}')
