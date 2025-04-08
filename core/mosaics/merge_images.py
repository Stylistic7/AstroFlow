import os
import ray
import numpy as np
from pathlib import Path
import astropy.io.fits as pyfits  # type: ignore
import tifffile

from context import context
from file_io.generic import read_header, read_image
from pix.classes import ImageList
from pix.decorators import PixInsight
from pix.tasks import PhotometricMosaic
from mosaics.types import Mosaic
from mosaics.artifacts import merged_filename
from previews import QualityLevel, autostretch
from task.astroflow_node import task
from file_io.generic import read_image, get_content_hash, update_header
from file_io.tiff_file_io import write_tiff_image


@PixInsight()
def run_photometric_mosaic(images: ImageList,
                           output_filename: str) -> ImageList:
  return PhotometricMosaic(images, output_filename)


@task.remote(resources={'PixInsight': 1})
def merge_images_with_photometric_mosaic(mosaic: Mosaic, input_0: Path,
                                         input_1: Path, filter: str) -> Path:
  print(f'### merge_images_with_photometric_mosaic {input_0} {input_1}')
  filename = merged_filename(mosaic, QualityLevel.MASTER,
                             get_content_hash(input_0),
                             get_content_hash(input_1))
  run_photometric_mosaic([input_0, input_1], filename)
  assert os.path.exists(filename), f'file not found: {filename}'
  update_header(filename, {'FILTER': filter})
  return filename


def merge_images_with_maximum(image_0: np.ndarray,
                              image_1: np.ndarray) -> np.ndarray:
  assert not np.isnan(np.min(image_0))
  assert not np.isnan(np.min(image_1))
  overlap = np.where((image_0 > 0) & (image_1 > 0))
  output_image = np.maximum(image_0, image_1)
  output_image[overlap] = (image_0[overlap] + image_1[overlap]) / 2.0
  assert not np.isnan(np.min(output_image))

  return output_image


@task.remote
def merge_images(mode: QualityLevel, mosaic: Mosaic, input_0: Path,
                 input_1: Path) -> Path:
  print(f'### merge images mode {mode} {input_0} {input_1}')

  if input_0 is None and input_1 is None:
    return None

  if input_0 is not None and input_1 is None:
    return input_0

  if input_0 is None and input_1 is not None:
    return input_1

  print(f'### merge_images input_0 header {read_header(input_0)}')
  print(f'### merge_images input_1 header {read_header(input_1)}')

  filters = set(read_header(i)['FILTER'] for i in [input_0, input_1])
  assert len(filters) == 1
  filter = next(iter(filters))

  # TODO: This is really ugly...
  if mode == QualityLevel.SUBFRAME:
    image_0 = autostretch(read_image(input_0))
    image_1 = autostretch(read_image(input_1))

    image = merge_images_with_maximum(image_0, image_1)

    output_filename = merged_filename(mosaic, mode, get_content_hash(input_0),
                                      get_content_hash(input_1))
    write_tiff_image(output_filename, image, {'FILTER': filter})

  elif mode == QualityLevel.MASTER:
    if os.environ.get('MOCK_PIXINSIGHT'):
      return input_0
    else:
      output_filename = ray.get(
          merge_images_with_photometric_mosaic.remote(mosaic, input_0, input_1,
                                                      filter))

  return output_filename
