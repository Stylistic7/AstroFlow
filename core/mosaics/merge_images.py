import os
import ray
import numpy as np
from pathlib import Path

from context import context
from pix.classes import ImageList
from pix.decorators import PixInsight
from pix.tasks import PhotometricMosaic
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
@AstroFlowNode(resources={'PixInsight': 1})
def merge_images_with_photometric_mosaic(input_0: Path, input_1: Path) -> Path:
  print(f'### merge_images_with_photometric_mosaic {input_0} {input_1}')

  image_0 = autostretch(read_image(input_0))
  image_1 = autostretch(read_image(input_1))
  assert not np.isnan(np.min(image_0))
  assert not np.isnan(np.min(image_1))
  overlap = np.where((image_0 > 0) & (image_1 > 0))
  assert np.count_nonzero(overlap)

  output_filename = Path(
      context.output_dir
  ) / 'Mosaics' / f'merged_{get_content_hash(input_0)}_{get_content_hash(input_1)}.xisf'

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
@AstroFlowNode(unpack_artifacts=False)
def merge_images(mode: QualityLevel, input_0: np.ndarray | Path,
                 input_1: np.ndarray | Path) -> np.ndarray | Path:
  print(f'### merge images {type(input_0)} {type(input_1)}')

  if input_0 is None and input_1 is None:
    return None

  if input_0 is not None and input_1 is None:
    return input_0

  if input_0 is None and input_1 is not None:
    return input_1

  # TODO: This is really ugly...
  if mode == 'subframe':
    return merge_images_with_maximum(input_0, input_1)
  elif mode == 'master':
    if context.mock_pixinsight:
      return input_0
    else:
      _, result = ray.get(
          merge_images_with_photometric_mosaic.remote(input_0, input_1))
      return result

  raise ValueError(f'Unknown quality level {mode}')
