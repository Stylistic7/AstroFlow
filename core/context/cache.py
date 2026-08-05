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


def _create_durable_cache() -> diskcache.Cache:
    """The never-evict store for not-cheaply-regenerable identity data
    (Stacks/Targets/Mosaics payloads + roots + config). Unlike the main
    cache, this is small in practice and MUST NOT evict: regenerating one
    of these means re-running global identification, so an LFU drop
    silently corrupts downstream lookups (the build_mosaic_subgraph
    None-stack crash). `eviction_policy='none'` => diskcache never culls;
    growth is bounded by reality (orphans from quarantine accrue very
    slowly; a reachability sweep can be added later if ever needed)."""
    if "ASTROFLOW_MOCK_CACHE" in os.environ:
        return diskcache.Cache(
            os.environ["ASTROFLOW_MOCK_CACHE"] + "_durable",
            eviction_policy="none",
            timeout=_CACHE_OP_TIMEOUT,
        )
    path = str(Path.home() / "Pictures" / "AstroFlow" / "durable")
    os.makedirs(path, exist_ok=True)
    return diskcache.Cache(
        path,
        size_limit=int(1024 * 1024 * 1024 * 1024),  # 1 TB; never culled anyway
        eviction_policy="none",
        timeout=_CACHE_OP_TIMEOUT,
    )


# Data that is NOT cheaply regenerable lives in the durable store: the
# identity/clustering (stacks/targets/mosaics/calibration_stacks) AND the
# user's curation (`decisions` — quarantines, calibration overrides,
# subframe-quality choices). Everything else — @node memo results,
# thumbnails, video, node_status, logs, ui_*, workflow_graphs — is
# regenerable on demand and stays in the evictable LFU cache, so it's safe
# to let it get evicted (or wiped).
_DURABLE_TUPLE_NS = frozenset(
    {"stacks", "targets", "mosaics", "calibration_stacks", "large_object",
     "decisions",
     # Per-display-product version history (graceful N/N-1 versioning). This
     # records PAST output paths/call_hashes, which nothing can recompute once
     # the old files are gone — an LFU eviction would silently lose the
     # last-good fallback, so it must never evict. See products/version_log.py.
     "product_versions",
     # Per-input-dir library scan records (fingerprint + listing). While a
     # volume is DISCONNECTED the record is the only knowledge of that disk's
     # contents ("disconnected = unchanged" semantics) — nothing can recompute
     # it until the disk returns, so it must never evict. Overwritten by a
     # fresh scan whenever the dir is present. See metadata/library_scan.py.
     "library_scan"}
)
_DURABLE_STR_KEYS = frozenset(
    {
        "stacks",
        "targets",
        "mosaics",
        "calibration_stacks",
        "settings",
        "input_dirs",
        "output_dir",
    }
)


def is_durable_key(key) -> bool:
    if isinstance(key, tuple) and key:
        return key[0] in _DURABLE_TUPLE_NS
    if isinstance(key, str):
        return key in _DURABLE_STR_KEYS
    return False


cache = None
durable_cache = None
queue_cache = None


def get_cache() -> diskcache.Cache:
    global cache

    if cache is None:
        cache = _create_cache()

    return cache


def get_durable_cache() -> diskcache.Cache:
    global durable_cache

    if durable_cache is None:
        durable_cache = _create_durable_cache()

    return durable_cache


def cache_for(key):
    """Route a key to the durable store or the evictable cache."""
    return get_durable_cache() if is_durable_key(key) else get_cache()


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
