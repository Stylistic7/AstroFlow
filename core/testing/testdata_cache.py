import tempfile
import tarfile
from dataclasses import replace
from pathlib import Path

import diskcache

from context import get_cache
from metadata.classes import ImageMetadata
from metadata.sidecar_io import read_image_metadata

TESTDATA_CACHE_GZIP_FILENAME = "testdata/cache.tar.gz"


def copy_testdata_into_cache():
    testdata_cache_dir = tempfile.TemporaryDirectory()

    with tarfile.open(TESTDATA_CACHE_GZIP_FILENAME, "r:gz") as tar:
        tar.extractall(path=testdata_cache_dir.name)

    testdata_cache = diskcache.Cache(testdata_cache_dir.name)

    cache = get_cache()
    num_copied = 0
    num_enriched = 0
    for key in testdata_cache.iterkeys():
        value = testdata_cache[key]
        # The cache.tar.gz predates the defect_candidates field. Enrich
        # ImageMetadata entries from the migrated sidecar so identify_stacks
        # doesn't exclude every legacy test frame as no-fingerprint.
        if (
            isinstance(value, ImageMetadata)
            and value.defect_candidates is None
            and Path(value.filename).exists()
        ):
            fresh = read_image_metadata(Path(value.filename))
            if fresh is not None and fresh.defect_candidates is not None:
                value = replace(
                    value, defect_candidates=fresh.defect_candidates
                )
                num_enriched += 1
        print(f"Copied key {key} into cache")
        cache[key] = value
        num_copied += 1

    print(f"### Copied {num_copied} keys from testdata cache "
          f"(enriched {num_enriched} with sidecar defect_candidates)")
