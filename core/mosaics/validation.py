import ray

from context import context
from mosaics.mosaic import MosaicsDict
from decorators.cached_decorator import cached


@cached.remote
def validate_mosaics(mosaics: MosaicsDict) -> None:
  print(f'### Entering validate_mosaics')
  targets = context.targets
  for mosaic_id, mosaic in mosaics.items():
    assert all(t in targets for t in mosaic.target_ids)

  return mosaics
