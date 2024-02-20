import os
import numpy as np
import objsize
import itertools
from xisf import XISF
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator
import largestinteriorrectangle
import astropy.io.fits as pyfits
import uuid
from spherical_geometry.polygon import SphericalPolygon
from shapely.geometry import Polygon, Point
import astropy.io.fits as pyfits

import file_io
import image_handling


def compute_viewbox(plate_solutions):
  ras = [f[0] for p in plate_solutions for f in p.footprint]
  decs = [f[1] for p in plate_solutions for f in p.footprint]
  # print(f'got {len(ras)} ras and {len(decs)} decs')

  min_ra, max_ra = np.min(ras), np.max(ras)
  min_dec, max_dec = np.min(decs), np.max(decs)

  return [min_ra, min_dec, max_ra - min_ra, max_dec - min_dec]


# TODO: Provide some feedback on progress.
def mosaic_by_coordinates(panel,
                          dest_xs,
                          dest_ys,
                          ras,
                          decs,
                          output_dir,
                          autostretch=False):
  print(f'### mosaic_by_coordinates projecting image {panel.image_filename}')

  projected_image = np.zeros(ras.shape)
  projection_map = np.zeros(ras.shape)
  print(f'### projected_image shape', projected_image.shape)

  # # TODO: Must also support XISF files here.
  # ext = os.path.splitext(panel.image_filename)[-1]
  # if ext in ['.fit', '.fits']:
  #   header = file_io.read_header(panel.image_filename)
  #   image = file_io.read_image(panel.image_filename)
  # elif ext == '.xisf':
  #   xisf = XISF(panel.image_filename)
  #   image = xisf.read_image(0).squeeze()
  #   img_meta = xisf.get_images_metadata()[0]
  #   header = {k: v[0]['value'] for k, v in img_meta['FITSKeywords'].items()}

  header = file_io.read_header(panel.image_filename)
  image = file_io.read_image(panel.image_filename)

  print(f'### mosaic_by_coordinates image dtype {image.dtype} '
        f'min {np.min(image)} max {np.max(image)}')

  if autostretch:
    try:
      image = image_handling.autostretch(image)
    except Exception as e:
      print(f'### failed to autostretch, exception {e}')

  print(f'### Image has shape {image.shape} and dtype {image.dtype}')

  wcs = WCS(panel.plate_solution.header)

  source_height, source_width = image.shape
  interp = RegularGridInterpolator((range(source_height), range(source_width)),
                                   image,
                                   method='linear',
                                   bounds_error=False,
                                   fill_value=-np.inf)

  # Filter out points that aren't in this panel's footprint.
  assert dest_xs.shape == dest_ys.shape == ras.shape == decs.shape
  footprint_polygon = SphericalPolygon.from_wcs(
      pyfits.Header(panel.plate_solution.header))
  print(f'### dest_xs.shape {dest_xs.shape} dest_ys.shape {dest_ys.shape}')
  print(f'### ras.shape {ras.shape} decs.shape {decs.shape}')
  dest_xs, dest_ys, ras, decs = zip(
      *[(dx, dy, ra, dec)
        for dx, dy, ra, dec in zip(dest_xs.flatten(), dest_ys.flatten(),
                                   ras.flatten(), decs.flatten())
        if footprint_polygon.contains_lonlat(ra, dec)])

  source_xs, source_ys = wcs.all_world2pix(ras, decs, 1, quiet=True)

  for dy, dx, sy, sx in zip(dest_ys, dest_xs, source_ys, source_xs):
    value = interp((sy, sx))
    if value == -np.inf:
      continue

    projected_image[dx, dy] = value
    projection_map[dx, dy] = 128

  print(f'Finished projecting image {panel.image_filename}')
  assert np.count_nonzero(projection_map), "projection failed"

  # TODO: This is surprisingly slow, dominating the projection for fine
  # output resolutions.
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
  print(f'### mosaic_by_coordinates output_image dtype {output_image.dtype} '
        f'min {np.min(output_image)} max {np.max(output_image)}')

  hdu = pyfits.PrimaryHDU(output_image)
  hdu.header['FOCALLEN'] = float(header['FOCALLEN'])
  hdu.header['XPIXSZ'] = float(header['XPIXSZ'])
  hdu.header['YPIXSZ'] = float(header['YPIXSZ'])
  hdu.header['HISTORY'] = f"TrimMosaicTile.target: foo"
  hdu.header['HISTORY'] = f"TrimMosaicTile.top: {trim_top}"
  hdu.header['HISTORY'] = f"TrimMosaicTile.bottom: {trim_top + trim_height}"
  hdu.header['HISTORY'] = f"TrimMosaicTile.left: {trim_left}"
  hdu.header['HISTORY'] = f"TrimMosaicTile.right: {trim_left + trim_width}"

  panel.image_filename = os.path.join(output_dir, f'placed_{uuid.uuid4()}.fit')
  hdu.writeto(panel.image_filename, overwrite=True)
  print(f'Wrote placed image filename {panel.image_filename}')

  return panel
