"""Per-dir library scan records: the edge-case matrix from
~/.claude/plans/library-fingerprint-instant-startup.md.

Pure tmp_path tests — the durable-record accessors are exercised against a
fake context (dict), so nothing here imports Ray.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from metadata import library_scan as ls


@pytest.fixture()
def fake_context(monkeypatch):
    store: dict = {}

    class _Ctx(dict):
        pass

    ctx = _Ctx()
    monkeypatch.setattr(ls, "_context", lambda: ctx)
    return ctx


def _mkfits(dir: Path, name: str, content: bytes = b"x" * 64) -> Path:
    p = dir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


# --- scan + fingerprint -------------------------------------------------------

def test_scan_is_deterministic(tmp_path):
    _mkfits(tmp_path, "a.fit")
    _mkfits(tmp_path, "b.fit")
    r1, r2 = ls.scan_dir(tmp_path), ls.scan_dir(tmp_path)
    assert r1.fingerprint == r2.fingerprint
    assert r1.listing == r2.listing


def test_fingerprint_changes_on_add_and_remove(tmp_path):
    _mkfits(tmp_path, "a.fit")
    r1 = ls.scan_dir(tmp_path)
    b = _mkfits(tmp_path, "b.fit")
    r2 = ls.scan_dir(tmp_path)
    assert r2.fingerprint != r1.fingerprint
    b.unlink()
    r3 = ls.scan_dir(tmp_path)
    assert r3.fingerprint == r1.fingerprint  # same content, same print


def test_fingerprint_changes_on_mtime_touch(tmp_path):
    a = _mkfits(tmp_path, "a.fit")
    r1 = ls.scan_dir(tmp_path)
    stamp = time.time() + 10
    os.utime(a, (stamp, stamp))
    assert ls.scan_dir(tmp_path).fingerprint != r1.fingerprint


# --- delta classes -------------------------------------------------------------

def test_delta_added_changed_removed(tmp_path):
    a = _mkfits(tmp_path, "a.fit")
    _mkfits(tmp_path, "b.fit")
    old = ls.scan_dir(tmp_path)

    _mkfits(tmp_path, "c.fit")                      # added
    stamp = time.time() + 5
    os.utime(a, (stamp, stamp))                     # changed
    (tmp_path / "b.fit").unlink()                   # removed
    new = ls.scan_dir(tmp_path)

    added, changed, removed = ls.delta(old, new)
    assert added == ["c.fit"]
    assert changed == ["a.fit"]
    assert removed == ["b.fit"]


def test_delta_first_scan_is_all_added(tmp_path):
    _mkfits(tmp_path, "a.fit")
    new = ls.scan_dir(tmp_path)
    added, changed, removed = ls.delta(None, new)
    assert added == ["a.fit"] and not changed and not removed


def test_delta_unchanged_is_empty(tmp_path):
    _mkfits(tmp_path, "a.fit")
    r = ls.scan_dir(tmp_path)
    assert ls.delta(r, ls.scan_dir(tmp_path)) == ([], [], [])


# --- record round-trip -----------------------------------------------------------

def test_record_round_trip(tmp_path, fake_context):
    _mkfits(tmp_path, "a.fit")
    rec = ls.scan_dir(tmp_path)
    ls.store_record(rec)
    back = ls.load_record(tmp_path)
    assert back == rec


def test_load_record_missing_or_malformed(tmp_path, fake_context):
    assert ls.load_record(tmp_path) is None
    fake_context[("library_scan", str(tmp_path))] = {"garbage": True}
    assert ls.load_record(tmp_path) is None


# --- disconnected semantics -------------------------------------------------------

_ABSENT_VOLUME_DIR = "/Volumes/__no_such_volume_astroflow_test__/Raw"


def test_dir_available_cases(tmp_path):
    assert ls.dir_available(tmp_path) is True
    # non-volume path that's simply missing = a real (empty) state
    assert ls.dir_available(tmp_path / "never_existed") is True
    # a dir on an UNMOUNTED volume = absent
    assert ls.dir_available(_ABSENT_VOLUME_DIR) is False


def test_dir_on_present_volume_but_deleted_is_a_real_change():
    volumes = [v for v in Path("/Volumes").iterdir() if v.is_dir()]
    if not volumes:
        pytest.skip("no mounted volumes")
    missing_dir = volumes[0] / "__astroflow_never_existed__"
    assert ls.dir_available(missing_dir) is True  # volume mounted → scan (empty)


def test_scan_or_reuse_present_is_fresh_and_does_not_persist(
    tmp_path, fake_context,
):
    _mkfits(tmp_path, "a.fit")
    rec, fresh = ls.scan_or_reuse(tmp_path)
    assert fresh is True and len(rec.listing) == 1
    # persistence is the PUBLISHER's job, after the pieces succeed
    assert ls.load_record(tmp_path) is None


def test_scan_or_reuse_disconnected_reuses_record(fake_context):
    known = ls.ScanRecord(
        dir=_ABSENT_VOLUME_DIR, fingerprint="fp-123",
        listing=(("a.fit", 64, 1), ("b.fit", 64, 2)),
    )
    ls.store_record(known)
    rec, fresh = ls.scan_or_reuse(_ABSENT_VOLUME_DIR)
    assert fresh is False
    assert rec == known  # verbatim: "this disk hasn't changed"


def test_scan_or_reuse_disconnected_never_scanned_is_empty(fake_context):
    rec, fresh = ls.scan_or_reuse(_ABSENT_VOLUME_DIR)
    assert fresh is False and rec.listing == ()


def test_reconnect_with_new_frames_yields_delta_only_for_them(
    tmp_path, fake_context,
):
    # "disk reconnects WITH new frames" from the matrix: old record vs a
    # fresh scan that has one extra file.
    _mkfits(tmp_path, "old.fit")
    old = ls.scan_dir(tmp_path)
    ls.store_record(old)
    _mkfits(tmp_path, "new.fit")
    rec, fresh = ls.scan_or_reuse(tmp_path)
    assert fresh
    added, changed, removed = ls.delta(ls.load_record(tmp_path), rec)
    assert added == ["new.fit"] and not changed and not removed
