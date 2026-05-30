"""Unit tests for the two-tier cart helper.

`apply_remove_from_cart` traces the pipeline at archive time to populate
the archive entry's paths (replacing the old per-cart-item snapshot
mechanism). These tests monkeypatch `paths_for_cart_item` so the cart→
archive state transitions can be exercised without a real pipeline.
"""
from __future__ import annotations

import pytest

from context import local_context as context
from state import archive


ITEM_A = ("mosaic", "abc", "PixInsight Master")
ITEM_B = ("stack", "def", "PixInsight Downscaled")


@pytest.fixture(autouse=True)
def _clean_cart_and_archive():
    """Reset both lists between cases."""
    original_cart = context.shopping_cart
    original_archive = context.archive
    context.shopping_cart = []
    context.archive = []
    yield
    context.shopping_cart = original_cart
    context.archive = original_archive


@pytest.fixture
def fake_trace(monkeypatch):
    """Stub `paths_for_cart_item` so archive transitions get fixed,
    inspectable path lists without needing a real pipeline. Returns
    the dict the test mutates to control what each item resolves to:
    `{item: {"all": [...], "finals": [...]}}`."""
    bank: dict[tuple, dict[str, list[str]]] = {}

    def fake(item, *, finals_only=False):
        entry = bank.get(tuple(item), {})
        key = "finals" if finals_only else "all"
        return [p for p in entry.get(key, [])]

    monkeypatch.setattr(archive, "paths_for_cart_item", fake)
    return bank


def test_remove_with_preserve_none_drops_cart(fake_trace):
    """preserve='none' clears the cart entry and creates no archive
    entry (no trace needed)."""
    fake_trace[ITEM_A] = {"all": ["/a"], "finals": []}
    context.shopping_cart = [ITEM_A]

    archive.apply_remove_from_cart(ITEM_A, preserve="none")

    assert list(context.shopping_cart or []) == []
    assert list(context.archive or []) == []


def test_remove_with_preserve_all_archives_all_traced_paths(fake_trace):
    """preserve='all' moves every traced path into the archive entry."""
    paths = ["/out/master.xisf", "/out/intermediate.xisf"]
    fake_trace[ITEM_A] = {"all": paths, "finals": ["/out/master.xisf"]}
    context.shopping_cart = [ITEM_A]

    archive.apply_remove_from_cart(ITEM_A, preserve="all")

    arch = list(context.archive or [])
    assert len(arch) == 1
    assert tuple(arch[0][:3]) == ITEM_A
    assert list(arch[0][3]) == paths


def test_remove_with_preserve_finals_archives_only_finals(fake_trace):
    finals = ["/out/master.xisf"]
    fake_trace[ITEM_A] = {
        "all": ["/out/intermediate", *finals],
        "finals": finals,
    }
    context.shopping_cart = [ITEM_A]

    archive.apply_remove_from_cart(ITEM_A, preserve="finals")

    arch = list(context.archive or [])
    assert len(arch) == 1
    assert list(arch[0][3]) == finals


def test_remove_with_unresolvable_item_skips_archive_entry(fake_trace):
    """If `paths_for_cart_item` returns [] (entity missing from
    context — stale cart entry), don't create an empty archive entry."""
    # No bank entry → trace returns [].
    context.shopping_cart = [ITEM_A]

    archive.apply_remove_from_cart(ITEM_A, preserve="all")

    assert list(context.archive or []) == []


def test_remove_replaces_existing_archive_entry(fake_trace):
    """Re-archiving an already-archived item replaces its preserved
    paths with the fresh trace rather than duplicating the entry."""
    new_paths = ["/out/new.xisf"]
    fake_trace[ITEM_A] = {"all": new_paths, "finals": []}
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
