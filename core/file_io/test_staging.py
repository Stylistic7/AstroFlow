"""Scratch-file reaping (`file_io.staging`).

A producer writes to `.{stem}.partial.{pid}.{tag}{suffix}` and publishes with
one `os.replace`, so leftover scratch is never the product: it is deleted once
its writer is dead and it is stale, and nothing else is ever touched.
"""
from __future__ import annotations

import os
import time

import pytest

from file_io.staging import DELETE, SKIP, scan, sweep

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


def test_stale_scratch_and_its_pyramid_are_deleted_and_the_product_kept(root):
    d = root / "an_op"
    scratch = _mk(d, f".abc123.partial.{_DEAD_PID}.0123abcd.tiff")
    pyr = _mk(d, f".abc123.partial.{_DEAD_PID}.0123abcd.tiff.pyr")
    product = _mk(d, "abc123.tiff", b"current")

    assert {c.path: c.action for c in scan([root])} == {scratch: DELETE, pyr: DELETE}
    sweep([root], dry_run=False)
    assert not scratch.exists() and not pyr.exists()
    assert product.read_bytes() == b"current"


def test_partial_copy_is_deleted(root):
    """publish_atomically's `.tmp.` file is a half-written copy — the
    destination is untouched until the replace."""
    partial = _mk(root / "an_op", f"abc123.tiff.tmp.{_DEAD_PID}.0123abcd")
    (c,) = scan([root])
    assert c.action == DELETE
    sweep([root], dry_run=False)
    assert not partial.exists()


def test_live_writer_is_never_touched(root):
    """A live PID means SKIP — an in-flight write looks just like debris."""
    scratch = _mk(root / "an_op", f".abc123.partial.{os.getpid()}.0123abcd.tiff")
    (c,) = scan([root])
    assert c.action == SKIP and "alive" in c.reason
    sweep([root], dry_run=False)
    assert scratch.exists()


def test_recent_file_is_never_touched(root):
    """The age guard is what makes this safe on a shared filesystem, where a
    PID from another node can look dead here."""
    fresh = _mk(root / "an_op", f".abc123.partial.{_DEAD_PID}.0123abcd.tiff", age_s=5)
    (c,) = scan([root])
    assert c.action == SKIP and "min old" in c.reason
    sweep([root], dry_run=False)
    assert fresh.exists()


def test_dry_run_changes_nothing(root):
    scratch = _mk(root / "an_op", f".abc123.partial.{_DEAD_PID}.0123abcd.xisf")
    assert sweep([root], dry_run=True)["deleted"] == 1
    assert scratch.exists()


def test_ordinary_files_are_ignored(root):
    """Real products and their sidecars must never be classified."""
    d = root / "an_op"
    for name in ("abc123.tiff", "abc123_tiles.tiff", "abc123.tiff.json", ".DS_Store"):
        _mk(d, name)
    assert scan([root]) == []
