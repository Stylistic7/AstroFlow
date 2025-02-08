import os
import ray
import numpy as np
from pathlib import Path
import astropy.io.fits as pyfits  # type: ignore
from xisf import XISF  # type: ignore

from context import context
from pix.classes import ImageList
from pix.decorators import PixInsight
from pix.tasks import PhotometricMosaic
from mosaics.types import Mosaic
from mosaics.placement import trim_image
from previews import QualityLevel, autostretch
from dependency_graph.astroflow_node import AstroFlowNode
from file_io.generic import read_image, get_content_hash

_TRIM_IMAGE_MEMORY_REQUIREMENT = 12 * 1024 * 1024 * 1024  # 12 GB


@PixInsight()
def run_photometric_mosaic(images: ImageList,
                           output_filename: str) -> ImageList:
  return PhotometricMosaic(images, output_filename)


@AstroFlowNode(memory=_TRIM_IMAGE_MEMORY_REQUIREMENT)
def trim_xisf_image(filename: Path) -> Path:
  xisf = XISF(filename)
  file_meta = xisf.get_file_metadata()
  file_meta
  ims_meta = xisf.get_images_metadata()
  ims_meta
  im_data = xisf.read_image(0)
  im_data = np.expand_dims(trim_image(np.squeeze(im_data)), -1)
  XISF.write(filename,
             im_data,
             creator_app="AstroFlow",
             image_metadata=ims_meta[0],
             xisf_metadata=file_meta,
             codec='zstd',
             shuffle=True)
  return filename


# TODO: Consider what to name these images, since the AstroFlowNode decorator
# handles all of the content hashes now.
# TODO: Turn caching on.
@AstroFlowNode(resources={'PixInsight': 1})
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


# TODO: Make sure that every Ray task that calls a @PixInsight function also
# consumes a PixInsight resource.
@AstroFlowNode(cache_result=True)
def merge_images(mode: QualityLevel, mosaic: Mosaic, input_0: Path,
                 input_1: Path) -> Path:
  print(f'### merge images {input_0} {input_1}')

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

    image = merge_images_with_maximum(image_0, image_1)

    output_filename = Path(
        context.output_dir
    ) / 'Mosaics' / mosaic.name / f'merged_subframe_{get_content_hash(input_0)}_{get_content_hash(input_1)}.fits'

    hdu = pyfits.PrimaryHDU(image)
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(str(output_filename), overwrite=True)

    return output_filename

  elif mode == 'master':
    if os.environ.get('MOCK_PIXINSIGHT'):
      return input_0
    else:
      _, result = ray.get(
          merge_images_with_photometric_mosaic.remote(mosaic, input_0, input_1))
      return result

  raise ValueError(f'Unknown quality level {mode}')
