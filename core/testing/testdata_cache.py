import tempfile
import tarfile
import diskcache

from context import get_cache

TESTDATA_CACHE_GZIP_FILENAME = "testdata/cache.tar.gz"


def copy_testdata_into_cache():
    testdata_cache_dir = tempfile.TemporaryDirectory()

    with tarfile.open(TESTDATA_CACHE_GZIP_FILENAME, "r:gz") as tar:
        tar.extractall(path=testdata_cache_dir.name)

    testdata_cache = diskcache.Cache(testdata_cache_dir.name)

    cache = get_cache()
    num_copied = 0
    for key in testdata_cache.iterkeys():
        print(f"Copied key {key} into cache")
        cache[key] = testdata_cache[key]
        num_copied += 1

    print(f"### Copied {num_copied} keys from testdata cache")
