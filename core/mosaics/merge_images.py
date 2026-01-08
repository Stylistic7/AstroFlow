import os
import ray
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import shutil

from context import context
from file_io.generic import read_header, read_image
from file_io.conversion import convert_tiff_to_xisf
from pix.classes import ImageList
from pix.decorators import PixInsight
from pix.tasks import PhotometricMosaic
from mosaics.types import Mosaic
from mosaics.artifacts import merged_filename_tiff, merged_filename_xisf
from previews.types import QualityLevel
from decorators import cached
from file_io.generic import read_image, update_header
from file_io.tiff_file_io import write_tiff_image
from file_io.types import XisfFilePath, TiffFilePath
from products import makes_image


@dataclass(frozen=True)
class MergedImage:
  pass


@makes_image(MergedImage).as_subframe(merged_filename_tiff)
@cached.remote
def merge_images_with_maximum(mosaic: Mosaic, input_0: Path,
                              input_1: Path) -> TiffFilePath:
  input_0 = input_0.resolve(strict=True)
  input_1 = input_1.resolve(strict=True)

  filters = set(read_header(i)['FILTER'] for i in [input_0, input_1])
  assert len(filters) == 1
  filter = next(iter(filters))

  image_0 = read_image(input_0)
  image_1 = read_image(input_1)

  assert not np.isnan(np.min(image_0))
  assert not np.isnan(np.min(image_1))
  overlap = np.where((image_0 > 0) & (image_1 > 0))
  output_image = np.maximum(image_0, image_1)
  output_image[overlap] = (image_0[overlap] + image_1[overlap]) / 2.0
  assert not np.isnan(np.min(output_image))

  output_filename = merge_images_with_maximum.filename_fn(
      mosaic, input_0, input_1)
  write_tiff_image(Path(output_filename), output_image, {'FILTER': filter})

  return output_filename


@PixInsight(timeout=300)
def run_photometric_mosaic(images: ImageList,
                           output_filename: str) -> ImageList:
  return PhotometricMosaic(images, output_filename)


@cached.remote(resources={'PixInsight': 1})
def merge_images_with_photometric_mosaic(mosaic: Mosaic, input_0: Path,
                                         input_1: Path,
                                         filter: str) -> XisfFilePath:
  print(f'### merge_images_with_photometric_mosaic {input_0} {input_1}')
  filename = merge_images_with_pmm.filename_fn(mosaic, input_0, input_1)
  run_photometric_mosaic([input_0, input_1], filename)
  assert os.path.exists(filename), f'file not found: {filename}'
  update_header(filename, {'FILTER': filter})
  return filename


@makes_image(MergedImage).as_master(merged_filename_xisf)
@cached.remote
def merge_images_with_pmm(mosaic: Mosaic, input_0: Path,
                          input_1: Path) -> XisfFilePath:
  input_0 = input_0.resolve(strict=True)
  input_1 = input_1.resolve(strict=True)

  print(f'### merge_images_with_pmm {input_0=} header {read_header(input_0)}')
  print(f'### merge_images_with_pmm {input_1=} header {read_header(input_1)}')

  filters = set(read_header(i)['FILTER'] for i in [input_0, input_1])
  assert len(filters) == 1
  filter = next(iter(filters))

  # TODO: This is messy. Clean it up.
  if context.get('mock_pixinsight', False):
    output_filename = merge_images_with_pmm.filename_fn(mosaic, input_0,
                                                        input_1)
    if isinstance(input_0, TiffFilePath):
      print(f'### mocking PixInsight for merging; '
            f'converting {input_0=} to {output_filename=}')
      return convert_tiff_to_xisf(input_0, output_filename)
    else:
      print(f'### mocking PixInsight for merging; '
            f'converting {input_0=} to {output_filename=}')
      shutil.copy(input_0, output_filename)
      return output_filename
  else:
    return ray.get(
        merge_images_with_photometric_mosaic.remote(mosaic, input_0, input_1,
                                                    filter))


@ray.remote
def merge_images(mode: QualityLevel, mosaic: Mosaic, input_0: Path,
                 input_1: Path) -> Path:
  if mode == QualityLevel.SUBFRAME:
    return merge_images_with_maximum.remote(mosaic, input_0, input_1)
  else:
    return merge_images_with_pmm.remote(mosaic, input_0, input_1)