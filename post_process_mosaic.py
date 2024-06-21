import ray

from context import context


@ray.remote(resources={'PixInsight': 1}, scheduling_strategy="SPREAD")
def post_process_mosaic(mosaic_id):
  mosaic = context.mosaics[mosaic_id]
  print(f'### Entering post_process_mosaic for mosaic {mosaic}')
