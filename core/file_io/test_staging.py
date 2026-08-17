"""Staging-file recovery (`file_io.staging`).

The load-bearing test is `test_parked_payload_without_canonical_is_restored`:
a parked payload whose canonical name is missing IS the product, and deleting
it destroys the only copy. That happened for real (an 18.9 GB
`combine_broadband_rgb` payload, 2026-08-17), which is why this module restores
rather than reaps.
"""
from __future__ import annotations

import os
import time

import pytest

from file_io.staging import DELETE, RESTORE, SKIP, scan, sweep

_OLD = 10_000.0        # older than DEFAULT_MIN_AGE_S
_DEAD_PID = 999_999    # never a live pid on macOS/Linux


def _mk(d, name, body=b"payload", age_s=_OLD):
    p = d / name
    p.write_bytes(body)
    t = time.time() - age_s
    os.utime(p, (t, t))
    return p


@pytest.fixture
def root(tmp_path):
    (tmp_path / "an_op").mkdir()
    return tmp_path


def test_parked_payload_without_canonical_is_restored(root):
    """The one that matters: canonical missing -> the staging file IS the
    product. Restore it; never delete it."""
    d = root / "an_op"
    parked = _mk(d, f".abc123.staging.{_DEAD_PID}.tiff", b"the only copy")

    (c,) = scan([root])
    assert c.action == RESTORE
    assert c.target == d / "abc123.tiff"

    sweep([root], dry_run=False)
    assert not parked.exists()
    assert (d / "abc123.tiff").read_bytes() == b"the only copy"


def test_parked_payload_with_canonical_is_deleted(root):
    """Canonical present -> the parked copy is a stale duplicate."""
    d = root / "an_op"
    parked = _mk(d, f".abc123.staging.{_DEAD_PID}.tiff")
    canonical = _mk(d, "abc123.tiff", b"current")

    (c,) = [x for x in scan([root]) if x.path == parked]
    assert c.action == DELETE

    sweep([root], dry_run=False)
    assert not parked.exists()
    assert canonical.read_bytes() == b"current"   # untouched


def test_partial_copy_is_deleted(root):
    """publish_atomically's `.tmp.` file is a half-written copy — the
    destination is untouched until the replace, so it is never a payload."""
    d = root / "an_op"
    partial = _mk(d, f"abc123.tiff.tmp.{_DEAD_PID}.0123abcd")

    (c,) = scan([root])
    assert c.action == DELETE
    sweep([root], dry_run=False)
    assert not partial.exists()


def test_live_writer_is_never_touched(root):
    """A live PID means SKIP — we cannot distinguish an in-flight publish from
    debris, so we keep it."""
    d = root / "an_op"
    parked = _mk(d, f".abc123.staging.{os.getpid()}.tiff")

    (c,) = scan([root])
    assert c.action == SKIP and "alive" in c.reason
    sweep([root], dry_run=False)
    assert parked.exists()


def test_recent_file_is_never_touched(root):
    """The age guard is what makes this safe on a shared filesystem, where a
    PID from another node can look dead here."""
    d = root / "an_op"
    fresh = _mk(d, f".abc123.staging.{_DEAD_PID}.tiff", age_s=5)

    (c,) = scan([root])
    assert c.action == SKIP and "min old" in c.reason
    sweep([root], dry_run=False)
    assert fresh.exists()


def test_dry_run_changes_nothing(root):
    d = root / "an_op"
    parked = _mk(d, f".abc123.staging.{_DEAD_PID}.tiff")

    r = sweep([root], dry_run=True)

    assert r["restored"] == 1
    assert parked.exists()
    assert not (d / "abc123.tiff").exists()


def test_ordinary_files_are_ignored(root):
    """Real products and their sidecars must never be classified."""
    d = root / "an_op"
    for name in ("abc123.tiff", "abc123_tiles.tiff", "abc123.tiff.json",
                 ".DS_Store"):
        _mk(d, name)
    assert scan([root]) == []


def test_suffix_is_preserved_on_restore(root):
    """`needs_display_pyramid` and the header readers dispatch on extension, so
    a restored .xisf must not come back as anything else."""
    d = root / "an_op"
    _mk(d, f".deadbeef.staging.{_DEAD_PID}.xisf")
    sweep([root], dry_run=False)
    assert (d / "deadbeef.xisf").exists()
