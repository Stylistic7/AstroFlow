import os
import diskcache  # type: ignore
from pathlib import Path

if 'ASTROFLOW_MOCK_CACHE' in os.environ:
  print('### Using test cache')
  cache = diskcache.Cache()
else:
  print('### Using real cache')
  # TODO: This path may not be correct on platforms other than MacOS.
  path = str(Path.home() / "Pictures" / "AstroFlow" / "cache")
  os.makedirs(path, exist_ok=True)
  cache = diskcache.Cache(path,
                          size_limit=int(50 * 1024 * 1024 * 1024),
                          eviction_policy='least-frequently-used')
