import os
import diskcache

if 'ASTROFLOW_MOCK_CACHE' in os.environ:
  print('### Using test cache')
  cache = diskcache.Cache()
else:
  print('### Using real cache')
  cache = diskcache.Cache(os.path.join(os.getcwd(), '.cache'),
                          size_limit=int(50 * 1024 * 1024 * 1024),
                          eviction_policy='least-frequently-used')