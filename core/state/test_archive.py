"""Unit tests for the two-tier cart helper.

Snapshots are written directly to the LocalContext-backed
diskcache (the cart→archive flow reads them via
`runtime_snapshot.load_cart_snapshot`); this isolates the
state transitions from the real pipeline runtime.
"""
from __future__ import annotations

import pytest

from context import local_context as context
from dependency_graph.runtime_snapshot import (
    SNAPSHOT_KEY_PREFIX,
    clear_cart_snapshot,
)
from state import archive


ITEM_A = ("mosaic", "abc", "PixInsight Master")
ITEM_B = ("stack", "def", "PixInsight Downscaled")


@pytest.fixture(autouse=True)
def _clean_cart_and_archive():
    """Reset both lists and any test-key snapshots between cases."""
    original_cart = context.shopping_cart
    original_archive = context.archive
    context.shopping_cart = []
    context.archive = []
    clear_cart_snapshot(ITEM_A)
    clear_cart_snapshot(ITEM_B)
    yield
    context.shopping_cart = original_cart
    context.archive = original_archive
    clear_cart_snapshot(ITEM_A)
    clear_cart_snapshot(ITEM_B)


def _write_snapshot(item, *, all_paths, final_paths):
    """Write a snapshot directly under the diskcache key
    `store_cart_snapshot` would have used."""
    context[(SNAPSHOT_KEY_PREFIX,) + tuple(item)] = {
        "all": list(all_paths), "finals": list(final_paths),
    }


def test_remove_with_preserve_none_drops_cart_and_snapshot():
    """preserve='none' clears the cart entry AND the live
    snapshot (its paths become orphan-eligible)."""
    _write_snapshot(ITEM_A, all_paths=["/a"], final_paths=[])
    context.shopping_cart = [ITEM_A]

    archive.apply_remove_from_cart(ITEM_A, preserve="none")

    assert list(context.shopping_cart or []) == []
    assert list(context.archive or []) == []
    from dependency_graph.runtime_snapshot import load_cart_snapshot
    assert load_cart_snapshot(ITEM_A) is None


def test_remove_with_preserve_all_archives_full_snapshot():
    """preserve='all' moves the snapshot's `all` list into the
    archive entry."""
    paths = ["/out/master.xisf", "/out/intermediate.xisf"]
    _write_snapshot(ITEM_A, all_paths=paths, final_paths=["/out/master.xisf"])
    context.shopping_cart = [ITEM_A]

    archive.apply_remove_from_cart(ITEM_A, preserve="all")

    arch = list(context.archive or [])
    assert len(arch) == 1
    assert tuple(arch[0][:3]) == ITEM_A
    assert list(arch[0][3]) == paths


def test_remove_with_preserve_finals_archives_only_finals():
    finals = ["/out/master.xisf"]
    _write_snapshot(
        ITEM_A,
        all_paths=["/out/intermediate", *finals],
        final_paths=finals,
    )
    context.shopping_cart = [ITEM_A]

    archive.apply_remove_from_cart(ITEM_A, preserve="finals")

    arch = list(context.archive or [])
    assert len(arch) == 1
    assert list(arch[0][3]) == finals


def test_remove_with_no_snapshot_skips_archive_entry():
    """If the item has never run (no snapshot exists), there's
    nothing to preserve — don't create an empty archive entry."""
    context.shopping_cart = [ITEM_A]

    archive.apply_remove_from_cart(ITEM_A, preserve="all")

    assert list(context.archive or []) == []


def test_remove_with_empty_snapshot_skips_archive_entry():
    """Snapshot exists but is empty (pipeline started and was
    interrupted before any node completed). Don't archive
    nothing."""
    _write_snapshot(ITEM_A, all_paths=[], final_paths=[])
    context.shopping_cart = [ITEM_A]

    archive.apply_remove_from_cart(ITEM_A, preserve="all")

    assert list(context.archive or []) == []


def test_remove_replaces_existing_archive_entry():
    """Re-archiving an already-archived item replaces its
    preserved paths with the fresh snapshot rather than
    duplicating the entry."""
    new_paths = ["/out/new.xisf"]
    _write_snapshot(ITEM_A, all_paths=new_paths, final_paths=[])
    context.shopping_cart = [ITEM_A]
    context.archive = [(*ITEM_A, ["/out/old.xisf"])]

    archive.apply_remove_from_cart(ITEM_A, preserve="all")

    arch = list(context.archive or [])
    assert len(arch) == 1
    assert list(arch[0][3]) == new_paths


def test_remove_from_archive_drops_matching_entry():
    context.archive = [(*ITEM_A, ["/x"]), (*ITEM_B, ["/y"])]

    archive.apply_remove_from_archive(ITEM_A)

    arch = list(context.archive or [])
    assert len(arch) == 1
    assert tuple(arch[0][:3]) == ITEM_B


def test_move_archive_to_cart_drops_archive_entry_and_adds_to_cart():
    context.shopping_cart = []
    context.archive = [(*ITEM_A, ["/x"])]

    archive.apply_move_archive_to_cart(ITEM_A)

    assert list(context.shopping_cart or []) == [ITEM_A]
    assert list(context.archive or []) == []


def test_invalid_preserve_value_raises():
    with pytest.raises(ValueError):
        archive.apply_remove_from_cart(ITEM_A, preserve="bogus")
