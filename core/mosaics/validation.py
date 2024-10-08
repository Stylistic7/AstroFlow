import ray

from context import context
from mosaics.mosaic import MosaicsDict


@ray.remote
def validate_mosaics(ancestor_results: list[MosaicsDict]) -> None:
  print(f'### Entering validate_mosaics')
  mosaics = ancestor_results[0]

  targets = context.targets
  for mosaic_id, mosaic in mosaics.items():
    assert all(t in targets for t in mosaic.target_ids)

  return mosaics
