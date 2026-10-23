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
    k = vl.key("mosaic", "abc123", "PI Master + OSS Mosaic", "Color", "RGB")
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


# ── supersede: the deterministic write-anchored N-1 deletion ──────────────

def _fake_log(monkeypatch, entries):
    """Route versions()/set_versions() to an in-memory dict (no context)."""
    store = {"log": list(entries)}
    monkeypatch.setattr(vl, "versions", lambda k: list(store["log"]))
    monkeypatch.setattr(
        vl, "set_versions", lambda k, es: store.__setitem__("log", list(es)))
    return store


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    return p


def test_supersede_deletes_other_versions_and_trims(tmp_path, monkeypatch):
    """When the new version lands, every OTHER logged version loses its
    primary + sidecar + pyramid, and the log becomes just the landed entry."""
    old = _touch(tmp_path / "op" / "aaaa1111.tiff")
    old_sidecar = _touch(tmp_path / "op" / "aaaa1111.tiff.json")
    old_pyr = _touch(tmp_path / "op" / "aaaa1111_tiles.tiff")
    landed = _touch(tmp_path / "op" / "bbbb2222.tiff")

    store = _fake_log(monkeypatch, [
        {"path": str(old), "call_hash": "aaaa1111", "ts": 0.0},
        {"path": str(landed), "call_hash": "bbbb2222", "ts": 1.0},
    ])
    k = vl.key("mosaic", "m1", "P", "Stars", "RGB")

    removed = vl.supersede(k, landed)

    assert removed == 3
    assert not old.exists() and not old_sidecar.exists() and not old_pyr.exists()
    assert landed.exists()
    assert [e["call_hash"] for e in store["log"]] == ["bbbb2222"]


def test_supersede_refuses_when_landed_file_missing(tmp_path, monkeypatch):
    """A completion that produced no file must never delete the last-good the
    UI is showing — supersede no-ops entirely."""
    old = _touch(tmp_path / "op" / "aaaa1111.tiff")
    store = _fake_log(monkeypatch, [
        {"path": str(old), "call_hash": "aaaa1111", "ts": 0.0},
    ])
    k = vl.key("mosaic", "m1", "P", "Stars", "RGB")

    removed = vl.supersede(k, tmp_path / "op" / "bbbb2222.tiff")

    assert removed == 0
    assert old.exists()
    assert [e["call_hash"] for e in store["log"]] == ["aaaa1111"]


def test_supersede_unchanged_rebuild_is_a_noop_delete(tmp_path, monkeypatch):
    """Landing the SAME call_hash again (cache-hit completion) deletes
    nothing — the landed entry is skipped by hash."""
    landed = _touch(tmp_path / "op" / "cccc3333.tiff")
    store = _fake_log(monkeypatch, [
        {"path": str(landed), "call_hash": "cccc3333", "ts": 0.0},
    ])
    k = vl.key("mosaic", "m1", "P", "Stars", "RGB")

    removed = vl.supersede(k, landed)

    assert removed == 0
    assert landed.exists()
    assert [e["call_hash"] for e in store["log"]] == ["cccc3333"]


def test_key_normalizes_object_type_case():
    """Writers use class names ("Mosaic", via extract_pipeline_taps); legacy
    URL routes pass lowercase ("mosaic"). Both must hit the same durable
    entries — the mismatch silently disabled the dashboard-side N-1
    fallback (2026-08-10)."""
    from products import version_log
    assert version_log.key("mosaic", "i", "p", "d", "H") == \
        version_log.key("Mosaic", "i", "p", "d", "H")
    assert version_log.key("stack", "i", "p", "d", None) == \
        version_log.key("Stack", "i", "p", "d", None)
