import ray

import os
import diskcache  # type: ignore
from pathlib import Path
from cache.lazy_load_dict import add_lazy_load_functionality

def _create_cache() -> diskcache.Cache:
  if 'ASTROFLOW_MOCK_CACHE' in os.environ:
    temp_dir = os.environ['ASTROFLOW_MOCK_CACHE']
    print(f'### Using test cache at {temp_dir}')
    cache = diskcache.Cache(temp_dir)
    return add_lazy_load_functionality(
      cache, ['stacks', 'targets', 'mosaics', 'calibration_stacks'])
  else:
    # TODO: This path may not be correct on platforms other than MacOS.
    path = str(Path.home() / "Pictures" / "AstroFlow" / "cache")
    print(f'### Using real cache at {path}')
    os.makedirs(path, exist_ok=True)
    cache = diskcache.Cache(path,
                           size_limit=int(50 * 1024 * 1024 * 1024),
                           eviction_policy='least-frequently-used')
    return add_lazy_load_functionality(
      cache, ['stacks', 'targets', 'mosaics', 'calibration_stacks'])


cache = None


def get_cache() -> diskcache.Cache:
  global cache

  if cache is None:
    cache = _create_cache()

  return cache
