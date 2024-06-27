import ray

from pix import PixInsight
from context import context
from decorators import saves_artifacts


@PixInsight
def _post_process_mosaic(image):
  return image


def post_process_mosaic_artifacts(mode, mosaic_id):
  mosaic = context.mosaics[mosaic_id]

  if mode == 'master':
    return {mosaic.content_hash: f'mosaic_{mosaic_id}_post_processed.fits'}


@saves_artifacts(post_process_mosaic_artifacts)
@ray.remote(resources={'PixInsight': 1}, scheduling_strategy="SPREAD")
def post_process_mosaic(mode, mosaic_id):
  mosaic = context.mosaics[mosaic_id]
  print(f'### Entering post_process_mosaic for mosaic {mosaic}')

  if mode == 'master':
    _, filename = next(
        iter(post_process_mosaic_artifacts(mode, mosaic_id).items()))
    return _post_process_mosaic(filename)
