import ray

import os
import diskcache  # type: ignore
from pathlib import Path


def _create_cache() -> diskcache.Cache:
  if 'ASTROFLOW_MOCK_CACHE' in os.environ:
    temp_dir = os.environ['ASTROFLOW_MOCK_CACHE']
    print(f'### Using test cache at {temp_dir}')
    return diskcache.Cache(temp_dir)
  else:
    # TODO: This path may not be correct on platforms other than MacOS.
    path = str(Path.home() / "Pictures" / "AstroFlow" / "cache")
    print(f'### Using real cache at {path}')
    os.makedirs(path, exist_ok=True)
    return diskcache.Cache(path,
                           size_limit=int(50 * 1024 * 1024 * 1024),
                           eviction_policy='least-frequently-used')


cache = None


def get_cache() -> diskcache.Cache:
  global cache

  if cache == None:
    cache = _create_cache()

  return cache
