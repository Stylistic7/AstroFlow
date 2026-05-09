import ray

import os
import diskcache  # type: ignore
from pathlib import Path


_CACHE_OP_TIMEOUT = 86400  # 1 day. Effectively "block until done."


def _create_cache() -> diskcache.Cache:
    if "ASTROFLOW_MOCK_CACHE" in os.environ:
        temp_dir = os.environ["ASTROFLOW_MOCK_CACHE"]
        print(f"### Using test cache at {temp_dir}")
        return diskcache.Cache(temp_dir, timeout=_CACHE_OP_TIMEOUT)
    else:
        # TODO: This path may not be correct on platforms other than MacOS.
        path = str(Path.home() / "Pictures" / "AstroFlow" / "cache")
        print(f"### Using real cache at {path}")
        os.makedirs(path, exist_ok=True)
        return diskcache.Cache(
            path,
            size_limit=int(100 * 1024 * 1024 * 1024),
            eviction_policy="least-frequently-used",
            # Default 60s SQL timeout caused spurious diskcache.Timeout
            # under heavy WBPP-driven concurrent writes (a single big
            # commit holding the lock long enough to starve any reader).
            # Silently swallowing those timeouts at the call site is a
            # correctness hazard for content-addressable lookups
            # (`('large_object', H)`), so we set the timeout high enough
            # that operations effectively block until completion. A real
            # deadlock will still surface eventually rather than masking.
            timeout=_CACHE_OP_TIMEOUT,
        )


cache = None
queue_cache = None


def get_cache() -> diskcache.Cache:
    global cache

    if cache is None:
        cache = _create_cache()

    return cache


def get_queue_cache() -> diskcache.Cache:
    global queue_cache

    if queue_cache is None:
        if "ASTROFLOW_MOCK_CACHE" in os.environ:
            temp_dir = os.environ["ASTROFLOW_MOCK_CACHE"] + "_queue"
            queue_cache = diskcache.Cache(temp_dir, timeout=_CACHE_OP_TIMEOUT)
        else:
            path = str(Path.home() / "Pictures" / "AstroFlow" / "queue_cache")
            os.makedirs(path, exist_ok=True)
            queue_cache = diskcache.Cache(path, timeout=_CACHE_OP_TIMEOUT)

    return queue_cache
