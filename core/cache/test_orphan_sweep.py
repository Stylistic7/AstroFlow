"""Unit tests for `cache.orphan_sweep.sweep` in the Phase 1b world.

Liveness is the RUNTIME CENSUS — the durable record of every path the last
COMPLETE pass actually landed — never a trace prediction (predictions were
proven to diverge from runtime landings; a trace-based live set pinned
phantoms and condemned the runtime's real files).

Floors that keep a sweep from ever deleting live data:
  * no complete census → `live_set` raises; sweep aborts before deleting;
  * an EMPTY live set refuses to delete anything (misconfiguration signal);
  * version-log pins and census siblings (sidecar + pyramid) always live.

`get_durable_cache` and `_pinned_versions` are monkeypatched — tests never
read the real durable cache.
"""
import types

import pytest

import cache.orphan_sweep as orphan_sweep
from dependency_graph.trace_paths import TraceResolutionError


def _ctx(tmp_path, archive=()):
    out = tmp_path / "Processed"
    out.mkdir(exist_ok=True)
    return types.SimpleNamespace(
        shopping_cart=[],
        archive=list(archive),
        output_dir=str(out),
    )


def _fake_census(monkeypatch, census):
    import context.cache as cc
    monkeypatch.setattr(
        cc, "get_durable_cache",
        lambda: types.SimpleNamespace(get=lambda k, default=None:
                                      census if k == "runtime_live_census"
                                      else default),
    )


def _no_pins(monkeypatch):
    monkeypatch.setattr(orphan_sweep, "_pinned_versions", lambda: set())


def test_live_set_refuses_without_complete_census(tmp_path, monkeypatch):
    """No census, or an incomplete one → refuse loudly. Sweeping against
    anything but a complete runtime ledger is forbidden."""
    _no_pins(monkeypatch)
    ctx = _ctx(tmp_path)

    _fake_census(monkeypatch, None)
    with pytest.raises(TraceResolutionError):
        orphan_sweep.live_set(ctx)

    _fake_census(monkeypatch, {"paths": ["/x.tiff"], "complete": False})
    with pytest.raises(TraceResolutionError):
        orphan_sweep.live_set(ctx)


def test_live_set_from_census_archive_and_siblings(tmp_path, monkeypatch):
    """The live set = census paths + archive paths, each expanded with its
    sidecar + pyramid siblings, plus version-log pins."""
    _fake_census(monkeypatch, {
        "paths": [str(tmp_path / "Processed" / "op" / "aaaa.tiff")],
        "complete": True,
    })
    monkeypatch.setattr(
        orphan_sweep, "_pinned_versions",
        lambda: {str(tmp_path / "pinned.tiff")},
    )
    ctx = _ctx(tmp_path, archive=[
        ("mosaic", "M1", "P", [str(tmp_path / "Processed" / "kept.tiff")]),
    ])

    live = orphan_sweep.live_set(ctx)
    assert str(tmp_path / "Processed" / "op" / "aaaa.tiff") in live
    assert str(tmp_path / "Processed" / "op" / "aaaa.tiff.json") in live
    assert str(tmp_path / "Processed" / "op" / "aaaa_tiles.tiff") in live
    assert str(tmp_path / "Processed" / "kept.tiff") in live
    assert str(tmp_path / "pinned.tiff") in live


def test_sweep_deletes_orphans_sparing_census(tmp_path, monkeypatch):
    """Only files absent from the census (and pins/archive/siblings) are
    orphans; delete=True removes exactly those."""
    _no_pins(monkeypatch)
    out = tmp_path / "Processed"
    op = out / "op"
    op.mkdir(parents=True)
    live_file = op / "live00.tiff"
    live_file.write_bytes(b"live")
    live_pyramid = op / "live00_tiles.tiff"
    live_pyramid.write_bytes(b"pyramid")
    dead = op / "dead00.tiff"
    dead.write_bytes(b"orphan")

    _fake_census(monkeypatch, {"paths": [str(live_file)], "complete": True})
    ctx = _ctx(tmp_path)

    report = orphan_sweep.sweep(ctx, delete=True)
    assert live_file.exists()
    assert live_pyramid.exists()          # sibling of a census path
    assert not dead.exists()
    assert report["deleted"] == 1


def test_empty_live_set_refuses_to_delete(tmp_path, monkeypatch):
    """A complete-but-empty census must never license wiping the output
    tree — the defensive floor survives the Phase 1b rebase."""
    _no_pins(monkeypatch)
    out = tmp_path / "Processed"
    op = out / "op"
    op.mkdir(parents=True)
    (op / "something.tiff").write_bytes(b"data")

    _fake_census(monkeypatch, {"paths": [], "complete": True})
    ctx = _ctx(tmp_path)

    report = orphan_sweep.sweep(ctx, delete=True)
    assert (op / "something.tiff").exists()
    assert report.get("skipped_empty_live") is True
    assert report["deleted"] == 0


def test_files_newer_than_census_are_spared(tmp_path, monkeypatch):
    """A file that LANDED AFTER the census postdates the deletion
    authority's knowledge (the R merge regenerated minutes after the first
    census, 2026-08-11) — it must classify LIVE even though the census has
    never heard of it. Pre-census orphans still die."""
    import os
    import time
    _no_pins(monkeypatch)
    out = tmp_path / "Processed"
    op = out / "op"
    op.mkdir(parents=True)

    past = time.time() - 3600
    old_orphan = op / "old_orphan.tiff"
    old_orphan.write_bytes(b"pre-census garbage")
    os.utime(old_orphan, (past, past))

    anchor = op / "anchor.tiff"
    anchor.write_bytes(b"census anchor")
    os.utime(anchor, (past, past))

    fresh = op / "fresh_landing.tiff"
    fresh.write_bytes(b"landed after the census")

    # Census taken 60s ago: knows only the anchor; ts guards the fresh file.
    _fake_census(monkeypatch, {
        "paths": [str(anchor)], "complete": True, "ts": time.time() - 60,
    })

    orphan_sweep.sweep(_ctx(tmp_path), delete=True)

    assert not old_orphan.exists(), "pre-census orphan must be deleted"
    assert fresh.exists(), "post-census landing must be SPARED"
    assert anchor.exists()
