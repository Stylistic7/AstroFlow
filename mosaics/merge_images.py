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


@pix.PixInsight()
def run_photometric_mosaic(images, output_filename):
  return pix.PhotometricMosaic(images, output_filename)


def merge_images_artifacts(mode, mosaic_id, input_0, input_1, output):
  print('### merge_images_artifacts, output', output)
  # TODO: Perhaps this should just be an Artifact?
  assert type(output) is Panel

  if mode == 'subframe':
    key = ('merged_images', 'subframe', mosaic_id, output.filter,
           output.merge_op_id, output.content_hash)
    return [Artifact(key=key)]
  elif mode == 'master':
    return [
        Artifact(filename=output.placed_filename,
                 content_hash=output.content_hash)
    ]


@ray.remote
@saves_artifacts(merge_images_artifacts)
def merge_images(mode, mosaic_id, input_0, input_1, output):
  print(f'### merge_images: inputs {input_0}, {input_1}')
  if mode == 'subframe':
    image_0 = autostretch(context[input_0.image_key])
    image_1 = autostretch(context[input_1.image_key])
    overlap = np.where((image_0 > 0) & (image_1 > 0))
    output_image = np.maximum(image_0, image_1)
    output_image[overlap] = (image_0[overlap] + image_1[overlap]) / 2.0

    key = next(
        iter(merge_images_artifacts(mode, mosaic_id, input_0, input_1,
                                    output))).key
    assert key
    context.set(key, output_image, flush=True)
    print(f'### Wrote key {key}')

    return key

  elif mode == 'master':
    return run_photometric_mosaic([input_0.filename, input_1.filename],
                                  output.filename)

  # filename = os.path.join(context.output_dir, output.filename)
  # image = np.random.random((10, 10))
  # hdu = pyfits.PrimaryHDU(image)
  # hdu.header['CONTHASH'] = output.content_hash
  # hdu.writeto(filename, overwrite=True)
