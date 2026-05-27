"""Unit tests for the cart-item path snapshot writer.

Hand-builds a `GraphRuntime`-shaped stub (just needs the
`filenames` dict) so we can verify the all/finals split without
spinning up a real pipeline.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from context import context
from dependency_graph.runtime_snapshot import (
    SNAPSHOT_KEY_PREFIX,
    clear_cart_snapshot,
    load_cart_snapshot,
    store_cart_snapshot,
)


KEY = ("mosaic", "abc", "PixInsight Master")


class _StubRuntime:
    """Minimal `GraphRuntime` stand-in — `store_cart_snapshot`
    only reads `.filenames`."""
    def __init__(self, filenames):
        self.filenames = filenames


def _make_node(name, *, final=False):
    """Synthetic Node carrying a `task` with `.task._final`
    matching `final` and a unique-enough hash."""
    def f():
        pass
    f.__name__ = name

    class Task:
        pass
    t = Task()
    t._function = f
    if final:
        t._final = True

    class Node:
        def __init__(self):
            self.task = t
            self.name = name

        def __hash__(self):
            return hash(name)

        def __eq__(self, other):
            return getattr(other, "name", None) == name
    return Node()


@pytest.fixture(autouse=True)
def _clean_snapshot():
    """Wipe any pre-existing snapshot under our test key before
    and after, so tests don't leak state."""
    clear_cart_snapshot(KEY)
    yield
    clear_cart_snapshot(KEY)


def test_store_splits_all_and_finals():
    intermediate = _make_node("place_image", final=False)
    finalized = _make_node("store_mosaic_color_channel_xisf", final=True)
    rt = _StubRuntime({
        intermediate: Path("/out/place.tiff"),
        finalized: Path("/out/final.xisf"),
    })

    store_cart_snapshot(KEY, rt)

    snap = load_cart_snapshot(KEY)
    assert snap is not None
    assert set(snap["all"]) == {"/out/place.tiff", "/out/final.xisf"}
    assert snap["finals"] == ["/out/final.xisf"]


def test_load_returns_none_for_unwritten_key():
    """Items that have never run have no snapshot. Loaders must
    not invent one."""
    assert load_cart_snapshot(("mosaic", "never", "P")) is None


def test_clear_removes_snapshot():
    rt = _StubRuntime({_make_node("op", final=False): Path("/x")})
    store_cart_snapshot(KEY, rt)
    assert load_cart_snapshot(KEY) is not None

    clear_cart_snapshot(KEY)

    assert load_cart_snapshot(KEY) is None


def test_store_overwrites_previous_snapshot():
    """Each pipeline run replaces the previous snapshot. Old
    paths from prior runs should not leak into the new snapshot
    (the runtime's `filenames` is a fresh dict each run)."""
    old_rt = _StubRuntime({_make_node("a"): Path("/old")})
    store_cart_snapshot(KEY, old_rt)
    new_rt = _StubRuntime({_make_node("b"): Path("/new")})
    store_cart_snapshot(KEY, new_rt)

    snap = load_cart_snapshot(KEY)
    assert snap["all"] == ["/new"]


def test_finals_can_be_empty():
    """A pipeline with no `final=True` ops still produces a
    snapshot — `finals` is just an empty list."""
    rt = _StubRuntime({
        _make_node("noop", final=False): Path("/out/x"),
    })
    store_cart_snapshot(KEY, rt)

    snap = load_cart_snapshot(KEY)
    assert snap["all"] == ["/out/x"]
    assert snap["finals"] == []
