"""Unit tests for the two-tier cart helper.

Patches `archive._resolve_paths` to bypass the real pipeline trace
— we want to verify the cart/archive state transitions, not the
trace itself (that's covered by test_trace_paths).
"""
from __future__ import annotations

import pytest

from context import local_context as context
from state import archive


ITEM_A = ("mosaic", "abc", "PixInsight Master")
ITEM_B = ("stack", "def", "PixInsight Downscaled")


@pytest.fixture(autouse=True)
def _clean_cart_and_archive():
    """Reset both lists before each test so they don't leak between
    cases."""
    original_cart = context.shopping_cart
    original_archive = context.archive
    context.shopping_cart = []
    context.archive = []
    yield
    context.shopping_cart = original_cart
    context.archive = original_archive


def _patch_paths(monkeypatch, paths_map):
    """Make `_resolve_paths(item, finals_only=...)` return whatever
    `paths_map[(item, finals_only)]` says — falling back to `[]`."""
    def fake(item, *, finals_only):
        return paths_map.get((tuple(item), finals_only), [])

    monkeypatch.setattr(archive, "_resolve_paths", fake)


def test_remove_with_preserve_none_drops_from_cart_only(monkeypatch):
    _patch_paths(monkeypatch, {})
    context.shopping_cart = [ITEM_A]

    archive.apply_remove_from_cart(ITEM_A, preserve="none")

    assert list(context.shopping_cart or []) == []
    assert list(context.archive or []) == []


def test_remove_with_preserve_all_appends_full_path_set(monkeypatch):
    paths = ["/out/master.xisf", "/out/intermediate.xisf"]
    _patch_paths(monkeypatch, {(ITEM_A, False): paths})
    context.shopping_cart = [ITEM_A]

    archive.apply_remove_from_cart(ITEM_A, preserve="all")

    assert list(context.shopping_cart or []) == []
    arch = list(context.archive or [])
    assert len(arch) == 1
    assert tuple(arch[0][:3]) == ITEM_A
    assert list(arch[0][3]) == paths


def test_remove_with_preserve_finals_appends_only_finals(monkeypatch):
    finals = ["/out/master.xisf"]
    _patch_paths(monkeypatch, {(ITEM_A, True): finals})
    context.shopping_cart = [ITEM_A]

    archive.apply_remove_from_cart(ITEM_A, preserve="finals")

    arch = list(context.archive or [])
    assert len(arch) == 1
    assert list(arch[0][3]) == finals


def test_remove_with_empty_path_set_skips_archive_entry(monkeypatch):
    """If the trace returns nothing (e.g. the item has never run
    so no on-disk files have content_hashes to feed into downstream
    call_hashes), don't create an empty archive entry."""
    _patch_paths(monkeypatch, {})
    context.shopping_cart = [ITEM_A]

    archive.apply_remove_from_cart(ITEM_A, preserve="all")

    assert list(context.archive or []) == []


def test_remove_replaces_existing_archive_entry(monkeypatch):
    """Re-archiving an already-archived item replaces its preserved
    paths with the fresh trace, rather than duplicating the entry."""
    old_paths = ["/out/old.xisf"]
    new_paths = ["/out/new.xisf"]
    _patch_paths(monkeypatch, {(ITEM_A, False): new_paths})
    context.shopping_cart = [ITEM_A]
    context.archive = [(*ITEM_A, old_paths)]

    archive.apply_remove_from_cart(ITEM_A, preserve="all")

    arch = list(context.archive or [])
    assert len(arch) == 1
    assert list(arch[0][3]) == new_paths


def test_remove_from_archive_drops_matching_entry(monkeypatch):
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


def test_invalid_preserve_value_raises(monkeypatch):
    _patch_paths(monkeypatch, {})
    with pytest.raises(ValueError):
        archive.apply_remove_from_cart(ITEM_A, preserve="bogus")
