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
        value = testdata_cache[key]
        # `defect_candidates` used to live on ImageMetadata; pre-migration
        # cache entries may carry it as a trailing pickle slot, but
        # ImageMetadata.__setstate__ truncates extra trailing entries so
        # the loaded value is just the up-to-date dataclass. The defects
        # themselves live in the sidecar (read separately by load_defects).
        print(f"Copied key {key} into cache")
        cache[key] = value
        num_copied += 1

    print(f"### Copied {num_copied} keys from testdata cache")
