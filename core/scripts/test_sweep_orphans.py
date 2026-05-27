"""End-to-end test of the sweep CLI's orphan classification.

Populates a temp output_dir with referenced + unreferenced files,
patches the cart/archive/output_dir state, and asserts that
`_walk_orphans` returns exactly the unreferenced files.

Doesn't exercise the actual pipeline tracing — `_live_set` calls
`paths_for_item`, which we stub with a known fixed return value to
isolate sweep classification from pipeline machinery.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from context import local_context as context
from scripts import sweep_orphans


@pytest.fixture
def temp_output_dir(tmp_path, monkeypatch):
    """Set `context.output_dir` to a fresh tmp dir for the test."""
    original = context.output_dir
    context.output_dir = str(tmp_path)
    yield tmp_path
    context.output_dir = original


@pytest.fixture(autouse=True)
def _clean_cart_and_archive():
    """Reset cart + archive around each test."""
    original_cart = context.shopping_cart
    original_archive = context.archive
    context.shopping_cart = []
    context.archive = []
    yield
    context.shopping_cart = original_cart
    context.archive = original_archive


def _make_file(path: Path, size: int = 100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def test_walk_orphans_returns_unreferenced_files_only(temp_output_dir):
    live_path = _make_file(temp_output_dir / "mosaic" / "live.xisf")
    orphan_path = _make_file(temp_output_dir / "mosaic" / "orphan.xisf")
    live = {str(live_path)}

    orphans = sweep_orphans._walk_orphans(temp_output_dir, live)
    paths = {p for p, _ in orphans}

    assert paths == {str(orphan_path)}


def test_live_set_includes_archive_paths_and_their_siblings(
    temp_output_dir, monkeypatch,
):
    """Archive entries' paths are live. Their sibling sidecar
    (`.xisf.json`) and tile pyramid (`.tiff`) are also live so
    the sweep doesn't strand them when a primary is preserved."""
    primary = temp_output_dir / "mosaic" / "preserved.xisf"
    sidecar = temp_output_dir / "mosaic" / "preserved.xisf.json"
    tile = temp_output_dir / "mosaic" / "preserved.tiff"
    _make_file(primary); _make_file(sidecar); _make_file(tile)

    # Bypass pipeline tracing — cart is empty, archive provides path.
    monkeypatch.setattr(sweep_orphans, "paths_for_item", lambda _i: [])
    context.archive = [("mosaic", "abc", "P", [str(primary)])]

    live = sweep_orphans._live_set()

    assert str(primary) in live
    assert str(sidecar) in live
    assert str(tile) in live


def test_live_set_includes_cart_traced_paths(
    temp_output_dir, monkeypatch,
):
    """Cart items contribute paths via `paths_for_item`."""
    live_path = temp_output_dir / "stack" / "master.xisf"
    _make_file(live_path)

    def fake_paths(item):
        if tuple(item) == ("stack", "xyz", "P"):
            return [live_path]
        return []

    monkeypatch.setattr(sweep_orphans, "paths_for_item", fake_paths)
    context.shopping_cart = [("stack", "xyz", "P")]
    context.archive = []

    live = sweep_orphans._live_set()

    assert str(live_path) in live


def test_full_sweep_classifies_correctly(temp_output_dir, monkeypatch):
    """End-to-end: cart references one master, archive preserves
    another, a third file is neither — only the third shows up
    as an orphan."""
    cart_path = _make_file(temp_output_dir / "a" / "cart.xisf")
    archive_path = _make_file(temp_output_dir / "b" / "archive.xisf")
    orphan_path = _make_file(temp_output_dir / "c" / "orphan.xisf")

    monkeypatch.setattr(
        sweep_orphans, "paths_for_item",
        lambda _i: [cart_path],
    )
    context.shopping_cart = [("stack", "1", "P")]
    context.archive = [("mosaic", "2", "P", [str(archive_path)])]

    live = sweep_orphans._live_set()
    orphans = sweep_orphans._walk_orphans(temp_output_dir, live)
    orphan_paths = {p for p, _ in orphans}

    assert orphan_paths == {str(orphan_path)}


def test_walk_orphans_returns_sizes(temp_output_dir):
    p = _make_file(temp_output_dir / "x.bin", size=512)
    orphans = sweep_orphans._walk_orphans(temp_output_dir, set())
    assert orphans == [(str(p), 512)]


def test_walk_orphans_with_no_output_dir_returns_empty(monkeypatch):
    """If output_dir is None (pre-init), sweep is a no-op."""
    assert sweep_orphans._walk_orphans(None, set()) == []
