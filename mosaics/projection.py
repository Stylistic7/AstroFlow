import ray
import os
import numpy as np
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator
import largestinteriorrectangle
import astropy.io.fits as pyfits
from spherical_geometry.polygon import SphericalPolygon
from shapely.geometry import Polygon, Point
import astropy.io.fits as pyfits
import time

from context import context
import file_io
import image_previews


def compute_viewbox(plate_solutions):
  ras = [f[0] for p in plate_solutions for f in p.footprint]
  decs = [f[1] for p in plate_solutions for f in p.footprint]
  # print(f'got {len(ras)} ras and {len(decs)} decs')

  min_ra, max_ra = np.min(ras), np.max(ras)
  min_dec, max_dec = np.min(decs), np.max(decs)

  return [min_ra, min_dec, max_ra - min_ra, max_dec - min_dec]


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
def mosaic_by_coordinates(panel):
  # TODO: Pass this through the context.
  autostretch = False

  image_hashes = context.image_hashes
  fits_headers = context.fits_headers
  plate_solutions = context.plate_solutions

  coordinates = context.get(('mosaic_coordinates', panel.mosaic_id))

  print(f'mosaic_by_coordinates projecting panel {panel}')
  start = time.time()

  dest_xs = ray.get(coordinates['dest_xs'])
  dest_ys = ray.get(coordinates['dest_ys'])
  ras = ray.get(coordinates['ras'])
  decs = ray.get(coordinates['decs'])

  projected_image = np.zeros(ras.shape)
  projection_map = np.zeros(ras.shape)
  print(f'projected_image shape', projected_image.shape)

  filename = image_hashes[panel.image_hash]
  header = fits_headers[panel.image_hash]
  plate_solution = plate_solutions[panel.image_hash]

  if header is None:
    print(f'Could not place panel {panel} because header was None')
    return context, None

  if plate_solution is None:
    print(f'Could not place panel {panel} because plate_solution was None')
    return context, None

  image = file_io.read_image(filename)

  print(f'mosaic_by_coordinates image dtype {image.dtype} '
        f'min {np.min(image)} max {np.max(image)}')

  if autostretch:
    try:
      image = image_previews.autostretch(image)
    except Exception as e:
      print(f'failed to autostretch, exception {e}')

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
  footprint_polygon = SphericalPolygon.from_wcs(
      pyfits.Header(plate_solution.header))
  print(f'dest_xs.shape {dest_xs.shape} dest_ys.shape {dest_ys.shape}')
  print(f'ras.shape {ras.shape} decs.shape {decs.shape}')
  try:
    dest_xs, dest_ys, ras, decs = zip(
        *[(dx, dy, ra, dec)
          for dx, dy, ra, dec in zip(dest_xs.flatten(), dest_ys.flatten(),
                                     ras.flatten(), decs.flatten())
          if footprint_polygon.contains_lonlat(ra, dec)])
  except Exception as e:
    print(f'### Failed to filter points with exception {e}')
    return context, None

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

  hdu = pyfits.PrimaryHDU(output_image)
  hdu.header['IMAGETYP'] = 'PlacedMosaic Master'
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
  print(f'Wrote placed image filename {panel.placed_filename}')

  return panel
