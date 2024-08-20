import os
import ray
import numpy as np
import astropy.io.fits as pyfits  # type: ignore
from pathlib import Path

from context import context
from artifacts import Artifact
import pix
from image_previews import autostretch
from artifacts.decorators import saves_artifacts
from artifacts import Artifact, read_artifact_image, write_artifact, artifact_exists


@pix.PixInsight()
def run_photometric_mosaic(images, output_filename):
  return pix.PhotometricMosaic(images, output_filename)


@ray.remote(resources={'PixInsight': 1})
def merge_images_with_photometric_mosaic(mode: str, mosaic_id: str,
                                         input_0: Artifact, input_1: Artifact,
                                         output: Artifact):
  image_0 = autostretch(read_artifact_image(input_0))
  image_1 = autostretch(read_artifact_image(input_1))
  assert not np.isnan(np.min(image_0))
  assert not np.isnan(np.min(image_1))
  overlap = np.where((image_0 > 0) & (image_1 > 0))
  if not np.count_nonzero(overlap):
    return [output]

  run_photometric_mosaic([input_0.filename, input_1.filename], output.filename)
  assert os.path.exists(output.filename)

  if output.content_hash:
    with pyfits.open(output.filename, mode='update') as hdu:
      header = hdu[0].header
      header['CONTHASH'] = output.content_hash
      hdu.flush()

  assert artifact_exists(output)

  return [output]


def merge_images_with_maximum(mode: str, mosaic_id: str, input_0: Artifact,
                              input_1: Artifact, output: Artifact):
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


def merge_images_artifacts(mode: str, mosaic_id: str, input_0: Artifact,
                           input_1: Artifact, output: Artifact):
  print('### merge_images_artifacts, output', output)
  return [output]


# TODO: Make sure that every Ray task that calls a @PixInsight function also
# consumes a PixInsight resource.
@ray.remote
@saves_artifacts(merge_images_artifacts)
def merge_images(mode: str, mosaic_id: str, input_0: Artifact,
                 input_1: Artifact, output: Artifact):
  print(f'### merge_images {mode}: inputs {input_0}, {input_1}')
  print(f'### merge_images {mode}: output {output}')

  assert artifact_exists(input_0), f'Input artifact {input_0} does not exist'
  assert artifact_exists(input_1), f'Input artifact {input_1} does not exist'

  if mode == 'subframe':
    result = merge_images_with_maximum(mode, mosaic_id, input_0, input_1,
                                       output)
  elif mode == 'master':
    result = ray.get(
        merge_images_with_photometric_mosaic.remote(mode, mosaic_id, input_0,
                                                    input_1, output))

  return result
