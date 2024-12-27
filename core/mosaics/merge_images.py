import os
import ray
import numpy as np
import astropy.io.fits as pyfits  # type: ignore
from pathlib import Path

from context import context
from artifacts import Artifact, FileArtifact, CacheArtifact
import pix
from pix.classes import ImageList, FilenameList
from pix.decorators import PixInsight
from pix.tasks import PhotometricMosaic
from previews import autostretch
from artifacts.decorators import saves_artifacts
from artifacts import Artifact, read_artifact_image, write_artifact, artifact_exists
from profiling import track_performance
from previews import QualityLevel
from mosaics.types import MosaicHash
from dependency_graph.logging import capture_logs
from dependency_graph.astroflow_node import AstroFlowNode
from mosaics import Mosaic


@PixInsight()
def run_photometric_mosaic(images: ImageList,
                           output_filename: str) -> ImageList:
  return PhotometricMosaic(images, output_filename)


@ray.remote(resources={'PixInsight': 1})
def merge_images_with_photometric_mosaic(input_0: Artifact,
                                         input_1: Artifact) -> Artifact:
  assert isinstance(input_0, FileArtifact)
  assert isinstance(input_1, FileArtifact)

  image_0 = autostretch(read_artifact_image(input_0))
  image_1 = autostretch(read_artifact_image(input_1))
  assert not np.isnan(np.min(image_0))
  assert not np.isnan(np.min(image_1))
  overlap = np.where((image_0 > 0) & (image_1 > 0))
  assert np.count_nonzero(overlap)

  output_filename = Path(
      context.output_dir
  ) / 'Mosaics' / f'merged_{input_0.content_hash}_{input_1.content_hash}.xisf'

  run_photometric_mosaic([input_0.filename, input_1.filename], output_filename)

  assert os.path.exists(output_filename)
  return output_filename


def merge_images_with_maximum(input_0: CacheArtifact,
                              input_1: CacheArtifact) -> np.ndarray:
  image_0 = autostretch(read_artifact_image(input_0))
  image_1 = autostretch(read_artifact_image(input_1))
  assert not np.isnan(np.min(image_0))
  assert not np.isnan(np.min(image_1))
  overlap = np.where((image_0 > 0) & (image_1 > 0))
  output_image = np.maximum(image_0, image_1)
  output_image[overlap] = (image_0[overlap] + image_1[overlap]) / 2.0
  assert not np.isnan(np.min(output_image))

  return output_image


# TODO: Make sure that every Ray task that calls a @PixInsight function also
# consumes a PixInsight resource.
#
# TODO: The @AstroFlowNode hides content hashes from functions that return
# np.ndarray or Path; perhaps it should also hide content hashes from functions
# that receive them?
@AstroFlowNode(unpack_artifacts=False)
def merge_images(mode: QualityLevel, input_0: Artifact,
                 input_1: Artifact) -> Artifact:
  print(f'### merge_images {mode}: inputs {input_0}, {input_1}')

  if mode == 'subframe':
    return merge_images_with_maximum(input_0, input_1)
  elif mode == 'master':
    return ray.get(merge_images_with_photometric_mosaic.remote(
        input_0, input_1))

  raise ValueError(f'Unknown quality level {mode}')
