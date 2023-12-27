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
                          plate_solution,
                          mosaic_dimensions,
                          output_dir,
                          autostretch=False):
  print(f'### Projecting image {panel.image_filename}')
  print(f'### panel has size '
        f'{objsize.get_deep_size(panel) / 1024.0 / 1024.0:.3f} MB')
  print(f'### plate_solution has size '
        f'{objsize.get_deep_size(plate_solution) / 1024.0 / 1024.0:.3f} MB')
  print(f'### mosaic_dimensions has size '
        f'{objsize.get_deep_size(mosaic_dimensions) / 1024.0 / 1024.0:.3f} MB')

  projected_image = np.zeros(
      (mosaic_dimensions.height, mosaic_dimensions.width))
  projection_map = np.zeros(
      (mosaic_dimensions.height, mosaic_dimensions.width))

  # TODO: Must also support XISF files here.
  ext = os.path.splitext(panel.image_filename)[-1]
  if ext in ['.fit', '.fits']:
    header = file_io.read_fits(panel.image_filename)
    image = file_io.read_fits_image(panel.image_filename)
  elif ext == '.xisf':
    xisf = XISF(panel.image_filename)
    image = xisf.read_image(0).squeeze()
    img_meta = xisf.get_images_metadata()[0]
    header = {k: v[0]['value'] for k, v in img_meta['FITSKeywords'].items()}

  if autostretch:
    image = image_handling.autostretch(image)

  print(f'### Image has shape {image.shape} and dtype {image.dtype}')
  print(f'### Projecting to dimensions '
        f'{mosaic_dimensions.height} x {mosaic_dimensions.width}')

  print('### FITS header is')
  for k, v in plate_solution.header.items():
    if 'COMMENT' in k: continue
    if 'HISTORY' in k: continue
    print(f'### {k}: {v}')

  wcs = WCS(plate_solution.header)
  # assert wcs.has_distortion

  source_height, source_width = image.shape
  interp = RegularGridInterpolator((range(source_height), range(source_width)),
                                   image,
                                   method='linear',
                                   bounds_error=False,
                                   fill_value=-np.inf)

  # TODO: Use projection other than Mercator.
  delta_ra = mosaic_dimensions.ra_resolution
  delta_dec = mosaic_dimensions.dec_resolution

  min_ra, min_dec, ra_span, dec_span = compute_viewbox([plate_solution])
  print(f'### This panel has min_ra {min_ra} ra_span {ra_span} '
        f'min_dec {min_dec} dec_span {dec_span}')
  max_ra = min_ra + ra_span
  max_dec = min_dec + dec_span

  start_ra_index = int(
      np.floor((min_ra - mosaic_dimensions.min_ra) /
               mosaic_dimensions.ra_resolution))
  end_ra_index = int(
      np.ceil((max_ra - mosaic_dimensions.min_ra) /
              mosaic_dimensions.ra_resolution))
  # print(f'### ra indices {start_ra_index} to {end_ra_index}')

  start_dec_index = int(
      np.floor((min_dec - mosaic_dimensions.min_dec) /
               mosaic_dimensions.dec_resolution))
  end_dec_index = int(
      np.ceil((max_dec - mosaic_dimensions.min_dec) /
              mosaic_dimensions.dec_resolution))
  # print(f'### dec indices {start_dec_index} to {end_dec_index}')

  dest_ys = np.arange(start_ra_index, end_ra_index)
  dest_xs = np.arange(start_dec_index, end_dec_index)

  ras = (mosaic_dimensions.min_ra + y * delta_ra for y in dest_ys)
  decs = (mosaic_dimensions.min_dec + x * delta_dec for x in dest_xs)
  ras, decs = zip(*itertools.product(ras, decs))
  # source_ys, source_xs = wcs.world_to_array_index_values(ras, decs)
  source_xs, source_ys = wcs.all_world2pix(ras, decs, 1)

  dest_ys, dest_xs = zip(*itertools.product(dest_ys, dest_xs))

  # print(f'### len(ras)', len(ras))
  # print(f'### ras[0:10]', ras[0:10])
  # print(f'### len(decs)', len(decs))
  # print(f'### decs[0:10]', decs[0:10])
  # print(f'### len(source_ys)', len(source_ys))
  # print(f'### source_ys[0:10]', source_ys[0:10])
  # print(f'### len(source_xs)', len(source_xs))
  # print(f'### source_xs[0:10]', source_xs[0:10])

  for dy, dx, sy, sx in zip(dest_ys, dest_xs, source_ys, source_xs):
    # print(f'### dy {dy} dx {dx} sy {sy} sx {sx}')
    value = interp((sy, sx))
    if value == -np.inf:
      continue

    projected_image[dy, dx] = value
    projection_map[dy, dx] = 128

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
                trim_height] = projected_image[trim_left:trim_left +
                                               trim_width,
                                               trim_top:trim_top + trim_height]

  output_image = trimmed_image

  hdu = pyfits.PrimaryHDU(output_image)
  hdu.header['FOCALLEN'] = float(header['FOCALLEN'])
  hdu.header['XPIXSZ'] = float(header['XPIXSZ'])
  hdu.header['YPIXSZ'] = float(header['YPIXSZ'])
  hdu.header['HISTORY'] = f"TrimMosaicTile.target: foo"
  hdu.header['HISTORY'] = f"TrimMosaicTile.top: {trim_top}"
  hdu.header['HISTORY'] = f"TrimMosaicTile.bottom: {trim_top + trim_height}"
  hdu.header['HISTORY'] = f"TrimMosaicTile.left: {trim_left}"
  hdu.header['HISTORY'] = f"TrimMosaicTile.right: {trim_left + trim_width}"

  for k, v in hdu.header.items():
    print(f'header {k}: {v}')

  panel.image_filename = os.path.join(output_dir, f'placed_{uuid.uuid4()}.fit')
  hdu.writeto(panel.image_filename, overwrite=True)
  print(f'Wrote placed image filename {panel.image_filename}')

  return panel