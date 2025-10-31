import tempfile
import tarfile
import diskcache

from context import get_cache

TESTDATA_CACHE_GZIP_FILENAME = 'testdata/cache.tar.gz'


def copy_testdata_into_cache():
  testdata_cache_dir = tempfile.TemporaryDirectory()

  with tarfile.open(TESTDATA_CACHE_GZIP_FILENAME, "r:gz") as tar:
    tar.extractall(path=testdata_cache_dir.name)

  testdata_cache = diskcache.Cache(testdata_cache_dir.name)

  cache = get_cache()
  num_copied = 0
  for key in testdata_cache.iterkeys():
    print(f'Copied key {key} into cache')
    cache[key] = testdata_cache[key]
    num_copied += 1

  print(f'### Copied {num_copied} keys from testdata cache')

  # TODO: These are a hack. Remove these when the testdata is refreshed next.
  # cache['stacks'] = testdata_cache['grouped_stacks']
  # cache['targets'] = testdata_cache['grouped_targets']
  # cache['mosaics'] = testdata_cache['grouped_mosaics']
  # print(f"### Copied {len(cache['stacks'])} stacks from testdata cache")
  # print(f"### Copied {len(cache['targets'])} targets from testdata cache")
  # print(f"### Copied {len(cache['mosaics'])} mosaics from testdata cache")
