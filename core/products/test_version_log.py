"""Pure unit tests for the per-product version log (Ray-free / cache-free).

Covers the two subtle correctness points of graceful N/N-1 versioning that are
easy to get wrong: the durable-store routing, the filter-key normalization
(so write-side raw `filter` and read-side string `filter` collide correctly),
and the dedup-or-clobber trim behavior.
"""
from products import version_log as vl


def _entry(call_hash: str) -> dict:
    return {"path": f"/out/op/{call_hash}.tiff", "call_hash": call_hash, "ts": 0.0}


def test_product_versions_routes_to_durable_store():
    """The log records PAST output paths — unrecoverable once the old files are
    swept — so its key MUST route to the never-evict durable store, else an LFU
    eviction silently loses the last-good fallback."""
    from context.cache import is_durable_key
    k = vl.key("mosaic", "abc123", "Hybrid: PI Master + OSS Mosaic", "Color", "RGB")
    assert k[0] == "product_versions"
    assert is_durable_key(k) is True


def test_key_normalizes_filter_including_none():
    """Write side passes raw `filter` (possibly None); read sides pass the tile
    URL string / dashboard `str(filter)`. Normalizing to str() makes them the
    same key — including the filterless case."""
    assert vl.key("stack", "s1", "P", "Master", "H") == \
           vl.key("stack", "s1", "P", "Master", "H")
    # None (filterless write) and the string "None" (a read) collapse together.
    assert vl.key("stack", "s1", "P", "Master", None) == \
           vl.key("stack", "s1", "P", "Master", "None")
    assert vl.key("stack", "s1", "P", "Master", None)[-1] == "None"


def test_call_hash_of_is_the_path_stem():
    assert vl.call_hash_of("/Volumes/Astro/Processed/blend/deadbeef.tiff") == "deadbeef"
    assert vl.call_hash_of("cafef00d.xisf") == "cafef00d"


def test_trimmed_append_grows_then_trims_to_keep():
    log: list = []
    log = vl._trimmed_append(log, _entry("a"))
    log = vl._trimmed_append(log, _entry("b"))
    assert [e["call_hash"] for e in log] == ["a", "b"]
    # A third distinct version trims the oldest — exactly N + N-1 survive.
    log = vl._trimmed_append(log, _entry("c"))
    assert [e["call_hash"] for e in log] == ["b", "c"]
    assert len(log) == vl._KEEP == 2


def test_trimmed_append_dedups_unchanged_rebuild():
    """The critical invariant: re-recording the SAME newest call_hash is a
    no-op — otherwise a duplicate append would trim the real N-1 away."""
    log = [_entry("a"), _entry("b")]
    same = vl._trimmed_append(log, _entry("b"))  # unchanged rebuild
    assert same is log  # exact no-op, list untouched
    assert [e["call_hash"] for e in same] == ["a", "b"]
