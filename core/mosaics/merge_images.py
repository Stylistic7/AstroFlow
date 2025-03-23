import os
import ray
import numpy as np
from pathlib import Path
import astropy.io.fits as pyfits  # type: ignore
from xisf import XISF  # type: ignore

from context import context
from file_io.generic import read_header
from pix.classes import ImageList
from pix.decorators import PixInsight
from pix.tasks import PhotometricMosaic
from mosaics.types import Mosaic
from previews import QualityLevel, autostretch
from dependency_graph.astroflow_node import AstroFlowNode
from file_io.generic import read_image, get_content_hash


@PixInsight()
def run_photometric_mosaic(images: ImageList,
                           output_filename: str) -> ImageList:
  return PhotometricMosaic(images, output_filename)


# TODO: Consider what to name these images, since the AstroFlowNode decorator
# handles all of the content hashes now.
# TODO: Turn caching on.
@AstroFlowNode(resources={'PixInsight': 1}, track_performance=True)
def merge_images_with_photometric_mosaic(mosaic: Mosaic, input_0: Path,
                                         input_1: Path) -> Path:
  print(f'### merge_images_with_photometric_mosaic {input_0} {input_1}')

  output_filename = Path(
      context.output_dir
  ) / 'Mosaics' / mosaic.name / f'merged_master_{get_content_hash(input_0)}_{get_content_hash(input_1)}.xisf'

  run_photometric_mosaic([input_0, input_1], output_filename)
  assert os.path.exists(output_filename)
  return output_filename


def merge_images_with_maximum(image_0: np.ndarray,
                              image_1: np.ndarray) -> np.ndarray:
  assert not np.isnan(np.min(image_0))
  assert not np.isnan(np.min(image_1))
  overlap = np.where((image_0 > 0) & (image_1 > 0))
  output_image = np.maximum(image_0, image_1)
  output_image[overlap] = (image_0[overlap] + image_1[overlap]) / 2.0
  assert not np.isnan(np.min(output_image))

  return output_image


# TODO: Cache these results.
@AstroFlowNode()
def merge_images(mode: QualityLevel, mosaic: Mosaic, input_0: Path,
                 input_1: Path) -> Path:
  print(f'### merge images {input_0} {input_1}')
  print(f'### merge_images type(input_0) {type(input_0)}')
  print(f'### merge_images type(input_1) {type(input_1)}')
  print(f'### merge_images Path(input_0) {Path(input_0)}')
  print(f'### merge_images Path(input_1) {Path(input_1)}')

  if input_0 is None and input_1 is None:
    return None

  if input_0 is not None and input_1 is None:
    return input_0

  if input_0 is None and input_1 is not None:
    return input_1

  # TODO: This is really ugly...
  if mode == 'subframe':
    image_0 = autostretch(read_image(input_0))
    image_1 = autostretch(read_image(input_1))

    filters = set(read_header(i)['FILTER'] for i in [input_0, input_1])
    assert len(filters) == 1
    filter = next(iter(filters))

    image = merge_images_with_maximum(image_0, image_1)

    output_filename = Path(
        context.output_dir
    ) / 'Mosaics' / mosaic.name / f'merged_subframe_{get_content_hash(input_0)}_{get_content_hash(input_1)}.fits'

    hdu = pyfits.PrimaryHDU(image)
    hdu.header['FILTER'] = filter
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(str(output_filename), overwrite=True)

  elif mode == 'master':
    if os.environ.get('MOCK_PIXINSIGHT'):
      return input_0
    else:
      _, output_filename = ray.get(
          merge_images_with_photometric_mosaic.remote(mosaic, input_0, input_1))

  os.remove(input_0)
  os.remove(input_1)

  return output_filename
