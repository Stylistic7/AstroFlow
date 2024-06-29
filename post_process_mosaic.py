import ray
from addict import Dict

from artifact import Artifact
from pix import PixInsight
from context import context
from decorators import saves_artifacts


@PixInsight
def _post_process_mosaic(image):
  return image


def post_process_mosaic_artifacts(mode, mosaic_id):
  mosaic = context.mosaics[mosaic_id]

  # TODO: What about filters?
  if mode == 'master':
    return [
        Artifact(content_hash=mosaic.content_hash,
                 filename=f'mosaic_{mosaic_id}_post_processed.fits')
    ]
  elif mode == 'subframe':
    return [
        Artifact(key=('mosaic_thumbnail_image', mosaic_id, mosaic.content_hash))
    ]


@ray.remote(resources={'PixInsight': 1}, scheduling_strategy="SPREAD")
@saves_artifacts(post_process_mosaic_artifacts)
def post_process_mosaic(mode, mosaic_id):
  mosaic = context.mosaics[mosaic_id]
  print(f'### Entering post_process_mosaic for mosaic {mosaic}')

  if mode == 'master':
    _, filename = next(
        iter(post_process_mosaic_artifacts(mode, mosaic_id).items()))
    return _post_process_mosaic(filename)
