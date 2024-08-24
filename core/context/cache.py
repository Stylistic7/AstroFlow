import ray

import os
import diskcache  # type: ignore
from pathlib import Path


def _create_cache():
  if 'ASTROFLOW_MOCK_CACHE' in os.environ:
    namespace = os.environ.get("ASTROFLOW_MOCK_NAMESPACE", "AstroFlow")
    print(f'### Using test cache in namespace {namespace}')
    manager = ray.get_actor('temp_dir_manager', namespace=namespace)
    temp_dir = ray.get(manager.get_temp_dir.remote())
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


def get_cache():
  global cache

  if cache == None:
    cache = _create_cache()

  return cache
