import os
import ray
import numpy as np
import astropy.io.fits as pyfits  # type: ignore

from artifact import Artifact
from .classes import Panel
import pix
from file_io.generic import read_image
from image_previews import autostretch, array_to_png
from decorators import saves_artifacts
from context import context
from artifact import Artifact, read_artifact_image, write_artifact


@pix.PixInsight()
def run_photometric_mosaic(images, output_filename):
  return pix.PhotometricMosaic(images, output_filename)


def merge_images_artifacts(mode: str, mosaic_id: str, input_0: Artifact,
                           input_1: Artifact, output: Artifact):
  print('### merge_images_artifacts, output', output)
  return [output]

  # if mode == 'subframe':
  #   key = ('merged_images', 'subframe', mosaic_id, output.filter,
  #          output.merge_op_id, output.content_hash)
  #   return [Artifact(key=key)]
  # elif mode == 'master':
  #   return [
  #       Artifact(filename=output.filename, content_hash=output.content_hash)
  #   ]


@ray.remote
@saves_artifacts(merge_images_artifacts)
def merge_images(mode: str, mosaic_id: str, input_0: Artifact,
                 input_1: Artifact, output: Artifact):
  print(f'### merge_images: inputs {input_0}, {input_1}')
  print(f'### merge_images: output {output}')
  if mode == 'subframe':
    image_0 = autostretch(read_artifact_image(input_0))
    image_1 = autostretch(read_artifact_image(input_1))
    overlap = np.where((image_0 > 0) & (image_1 > 0))
    output_image = np.maximum(image_0, image_1)
    output_image[overlap] = (image_0[overlap] + image_1[overlap]) / 2.0

    # key = next(
    #     iter(merge_images_artifacts(mode, mosaic_id, input_0, input_1,
    #                                 output))).key
    # assert key

    key = output.key

    filename = '/tmp/foo/' + '_'.join(key) + '.png'
    open(filename, 'wb').write(array_to_png(output_image))

    # context.set(key, output_image, flush=True)
    # print(f'### Wrote key {key}')
    # return key

    write_artifact(output, output_image)

  elif mode == 'master':
    return run_photometric_mosaic([input_0.filename, input_1.filename],
                                  output.filename)

  # filename = os.path.join(context.output_dir, output.filename)
  # image = np.random.random((10, 10))
  # hdu = pyfits.PrimaryHDU(image)
  # hdu.header['CONTHASH'] = output.content_hash
  # hdu.writeto(filename, overwrite=True)
