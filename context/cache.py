import os
import diskcache  # type: ignore
import traceback

if 'ASTROFLOW_MOCK_CACHE' in os.environ:
  print('### Using test cache')
  for line in traceback.format_stack():
    print(line.strip())
  cache = diskcache.Cache()
else:
  print('### Using real cache')
  for line in traceback.format_stack():
    print(line.strip())
  cache = diskcache.Cache(os.path.join(os.getcwd(), '.cache'),
                          size_limit=int(50 * 1024 * 1024 * 1024),
                          eviction_policy='least-frequently-used')
