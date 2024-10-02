import os
import ray
import numpy as np
import astropy.io.fits as pyfits  # type: ignore
from pathlib import Path

from context import context
from artifacts import Artifact, FileArtifact
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


@PixInsight()
def run_photometric_mosaic(images: ImageList,
                           output_filename: str) -> ImageList:
  return PhotometricMosaic(images, output_filename)


@ray.remote(resources={'PixInsight': 1})
def merge_images_with_photometric_mosaic(mode: QualityLevel,
                                         mosaic_id: MosaicHash,
                                         input_0: Artifact, input_1: Artifact,
                                         output: Artifact) -> list[Artifact]:
  image_0 = autostretch(read_artifact_image(input_0))
  image_1 = autostretch(read_artifact_image(input_1))
  assert not np.isnan(np.min(image_0))
  assert not np.isnan(np.min(image_1))
  overlap = np.where((image_0 > 0) & (image_1 > 0))
  if not np.count_nonzero(overlap):
    return [output]

  assert isinstance(input_0, FileArtifact)
  assert isinstance(input_1, FileArtifact)
  assert isinstance(output, FileArtifact)
  run_photometric_mosaic([input_0.filename, input_1.filename], output.filename)
  assert os.path.exists(output.filename)

  if output.content_hash:
    with pyfits.open(output.filename, mode='update') as hdu:
      header = hdu[0].header
      header['CONTHASH'] = output.content_hash
      hdu.flush()

  assert artifact_exists(output)

  return [output]


def merge_images_with_maximum(mode: QualityLevel, mosaic_id: MosaicHash,
                              input_0: Artifact, input_1: Artifact,
                              output: Artifact) -> list[Artifact]:
  image_0 = autostretch(read_artifact_image(input_0))
  image_1 = autostretch(read_artifact_image(input_1))
  assert not np.isnan(np.min(image_0))
  assert not np.isnan(np.min(image_1))
  overlap = np.where((image_0 > 0) & (image_1 > 0))
  output_image = np.maximum(image_0, image_1)
  output_image[overlap] = (image_0[overlap] + image_1[overlap]) / 2.0
  assert not np.isnan(np.min(output_image))

  write_artifact(output, output_image)

  return [output]


def merge_images_artifacts(mode: QualityLevel, mosaic_id: MosaicHash,
                           input_0: Artifact, input_1: Artifact,
                           output: Artifact) -> list[Artifact]:
  print('### merge_images_artifacts, output', output)
  return [output]


# TODO: Make sure that every Ray task that calls a @PixInsight function also
# consumes a PixInsight resource.
@ray.remote
@track_performance
@saves_artifacts(merge_images_artifacts)
def merge_images(mode: QualityLevel, mosaic_id: MosaicHash, input_0: Artifact,
                 input_1: Artifact, output: Artifact) -> list[Artifact]:
  print(f'### merge_images {mode}: inputs {input_0}, {input_1}')
  print(f'### merge_images {mode}: output {output}')

  assert artifact_exists(input_0), f'Input artifact {input_0} does not exist'
  assert artifact_exists(input_1), f'Input artifact {input_1} does not exist'

  if mode == 'subframe':
    return merge_images_with_maximum(mode, mosaic_id, input_0, input_1, output)
  elif mode == 'master':
    return ray.get(
        merge_images_with_photometric_mosaic.remote(mode, mosaic_id, input_0,
                                                    input_1, output))

  raise ValueError(f'Unknown quality level {mode}')
