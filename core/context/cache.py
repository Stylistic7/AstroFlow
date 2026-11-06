import ray

import os
import diskcache  # type: ignore
from pathlib import Path


# "Block until done" for the compute graph (a big WBPP commit can hold the
# SQLite lock long enough to starve readers, and silently-swallowed timeouts
# corrupt content-addressed lookups). Web workers override via env — a
# gunicorn thread must NEVER block for a day on the cache lock; it should
# error fast and let the client retry (see state/gunicorn_control.py).
_CACHE_OP_TIMEOUT = float(os.environ.get("ASTROFLOW_CACHE_TIMEOUT_S", 86400))


def _evictable_path() -> str:
    if "ASTROFLOW_MOCK_CACHE" in os.environ:
        return os.environ["ASTROFLOW_MOCK_CACHE"]
    # TODO: This path may not be correct on platforms other than MacOS.
    return str(Path.home() / "Pictures" / "AstroFlow" / "cache")


def _hosts_evictable_cache() -> bool:
    """Whether this process may open the evictable cache: only the node
    that carries the canonical one — the Ray head, marked by the head-only
    `diskcache` resource. Every other node's @node results, logs and
    concerns go through the ContextActor to the head's store; a worker has
    NO evictable cache of its own, and one it opens anyway is a silent fork
    of cluster state (the 2026-09-14 finding: a pickled cache handle gave
    the MacBook a private 170k-entry cache the head never saw). Without Ray
    (scripts, tests, a web worker before its lazy connect) there is nothing
    to be a worker of, so the answer is yes."""
    try:
        import ray
        if not ray.is_initialized():
            return True
        node_id = ray.get_runtime_context().get_node_id()
        return any(n["NodeID"] == node_id and "diskcache" in n.get("Resources", {})
                   for n in ray.nodes())
    except Exception:
        return True


def _create_cache() -> diskcache.Cache:
    if not _hosts_evictable_cache():
        raise RuntimeError(
            "the evictable diskcache is the head node's; a worker has none and "
            "must reach it through the ContextActor (decorators._head_aware_cache)")
    if "ASTROFLOW_MOCK_CACHE" in os.environ:
        temp_dir = _evictable_path()
        print(f"### Using test cache at {temp_dir}")
        return diskcache.Cache(temp_dir, timeout=_CACHE_OP_TIMEOUT)
    else:
        path = _evictable_path()
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


# Two reasons a key lives in the durable (non-evicting) store:
#
# 1. NOT cheaply regenerable: identity/clustering payloads
#    (stacks/targets/mosaics/calibration_stacks) and the user's CURATION
#    (decisions, shopping_cart, archive, observing_plan, device nicknames —
#    an LFU cull of any of these silently loses user work).
# 2. HOT WEB-READ PATH: the LFU policy turns every read into a SQLite
#    BEGIN IMMEDIATE (the access-count bump is a write transaction), so a
#    web read of an LFU key contends with the compute graph's writes on the
#    same 1.25 GB cache.db — the mechanism behind multi-minute web-server
#    wedges during rebuilds. `eviction_policy='none'` reads take diskcache's
#    lock-free fast path. Fixed-key, bounded-size web state (ui_*,
#    tap_snapshots) therefore lives here even though it IS regenerable.
#
# Everything else — @node memos, thumbnails, video, node_status, logs,
# workflow_graphs — is regenerable, unbounded-keyed, and stays in the
# evictable LFU cache.
_DURABLE_TUPLE_NS = frozenset(
    {"stacks", "targets", "mosaics", "calibration_stacks", "large_object",
     "decisions",
     # Saved finish looks, per (object type, id, pipeline, finish): user
     # curation -- an eviction loses the user's work. See
     # web_server/backend/finish_saves.py.
     "finish_saves",
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
     "library_scan",
     # Per-shared-folder record of the masters THIS instance published there
     # (`stacks.publish_shared`): what makes a file in a shared folder ours to
     # retire. Nothing else can tell — not a name, not a stable hash.
     "shared_published",
     # Tap-snapshot payloads (ContextRef targets for the str-keyed snapshot
     # index below) — hot web-read path, see reason 2 above.
     "tap_snapshots"}
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
        "shared_dirs",
        "astrometry_index_dir",
        "tool_paths",
        # --- user curation (reason 1: an eviction loses user work) ---
        "shopping_cart",
        "observing_plan",
        "asiair_cooling",
        # --- hot web-read state (reason 2: keep reads off the LFU lock) ---
        "ui_stacks",
        "ui_targets",
        "ui_mosaics",
        "ui_calibration_stacks",
        "tap_snapshots",
        "node_running",
        # --- machine configuration (reason 1 again: the 2026-08-06 wipe
        # silently reverted cluster_interface to auto-detect → head bound the
        # WiFi NIC (10.0.0.x) → the MBP could not join → split-brain with two
        # clusters double-processing the shared volumes. Network settings are
        # user CONFIGURATION, never derived state — they must survive
        # eviction and wipes. ---
        "network_settings",
        "role_chosen",
    }
)

# Keys re-routed from the evictable store on 2026-08-05: copy any value still
# sitting in the old store so curation isn't silently lost and the UI isn't
# blank on first boot after the change. One direct get per key (cheap);
# existing durable values always win. tap-snapshot PAYLOADS (tuple ns) are
# deliberately not migrated — they self-heal per pipeline run via
# _store_tap_snapshot's presence check.
_MIGRATE_STR_KEYS = (
    "shopping_cart", "observing_plan",
    "ui_stacks", "ui_targets", "ui_mosaics", "ui_calibration_stacks",
    "tap_snapshots",
    "network_settings", "role_chosen",
)


def _migrate_rerouted_keys(durable: diskcache.Cache) -> None:
    """Opens the old evictable store directly (not `get_cache`, which a worker
    may not call): a worker's machine configuration was written there before
    `network_settings` routed it durable, and this is how it comes across."""
    path = _evictable_path()
    if not os.path.isdir(path):
        return
    try:
        main = diskcache.Cache(path, timeout=_CACHE_OP_TIMEOUT)
    except Exception as e:  # migration must never block startup
        print(f"### migration source could not open (non-fatal): {e}")
        return
    with main:
        for key in _MIGRATE_STR_KEYS:
            try:
                if key in durable:
                    continue
                value = main.get(key, None)
                if value is not None:
                    durable[key] = value
                    print(f"### migrated {key!r} from evictable to durable store")
            except Exception as e:
                print(f"### migration of {key!r} failed (non-fatal): {e}")


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
        _migrate_rerouted_keys(durable_cache)

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
