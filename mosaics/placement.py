import ray
import os
import numpy as np
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator
import largestinteriorrectangle
import time

from plate_solving import solve_field
from decorators import saves_artifacts
from artifact import Artifact
from context import context
from .viewbox import compute_mosaic_viewbox_artifacts
from artifact import read_artifact_image, read_artifact_header, write_artifact, CacheArtifact, FileArtifact


# TODO: This needs to include the hash of the viewbox.
def mosaic_by_coordinates_artifacts(mode, mosaic_id, stack_id, input_artifact):
  if mode == 'subframe':
    key = ('placed_image', mosaic_id, stack_id)
    return [CacheArtifact(key, input_artifact.content_hash)]
  elif mode == 'master':
    mosaic = context.mosaics[mosaic_id]
    filename = os.path.join(context.output_dir, 'Mosaics', mosaic.name,
                            f'placed_{mosaic_id}_{stack_id}.fit')
    return [FileArtifact(filename, input_artifact.content_hash)]


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

  viewbox_key = next(iter(compute_mosaic_viewbox_artifacts(mode,
                                                           mosaic_id))).key
  viewbox = context.get(viewbox_key)
  assert viewbox

  # TODO: Remove the dest_x and dest_y fields from the viewbox.
  ras = viewbox['ras']
  decs = viewbox['decs']

  projected_image = np.zeros(ras.shape)

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

  if plate_solution is None:
    print(f'Could not place artifact {input_artifact} because plate_solution '
          f'was None')
    return

  print(f'mosaic_by_coordinates: image shape {image.shape} dtype {image.dtype} '
        f'min {np.min(image)} max {np.max(image)}')

  wcs = WCS(plate_solution.header)

  source_height, source_width = image.shape
  interp = RegularGridInterpolator((range(source_height), range(source_width)),
                                   image,
                                   method='linear',
                                   bounds_error=False,
                                   fill_value=np.nan)

  source_xs, source_ys = wcs.all_world2pix(ras, decs, 1, quiet=True)
  projected_image = interp((source_ys, source_xs))

  # TODO: This is surprisingly slow, dominating the projection for fine
  # output resolutions.
  trim_top, trim_left, trim_height, trim_width = largestinteriorrectangle.lir(
      projected_image != np.nan)
  print(f'Trimmed trim_top {trim_top} trim_left {trim_left} trim_height '
        f'{trim_height} trim_width {trim_width}')

  projected_image = np.nan_to_num(projected_image)

  trimmed_image = np.zeros(projected_image.shape, np.float32)
  trimmed_image[trim_left:trim_left + trim_width, trim_top:trim_top +
                trim_height] = projected_image[trim_left:trim_left + trim_width,
                                               trim_top:trim_top + trim_height]

  output_image = trimmed_image
  print(f'### mosaic_by_coordinates: trimming '
        f'output_image dtype {output_image.dtype} '
        f'min {np.min(output_image)} max {np.max(output_image)}')

  output_artifact = next(
      iter(
          mosaic_by_coordinates_artifacts(mode, mosaic_id, stack_id,
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
        'HISTORY': f"TrimMosaicTile.bottom: {trim_top + trim_height}",
        'HISTORY': f"TrimMosaicTile.left: {trim_left}",
        'HISTORY': f"TrimMosaicTile.right: {trim_left + trim_width}"
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
