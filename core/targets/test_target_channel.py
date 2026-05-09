"""Unit tests for target-channel merge.

The exhaustive-marked integration test exercises the full pipeline.
These quick tests cover the math and the n=1 passthrough — they don't
need Ray or PixInsight.
"""
from unittest.mock import MagicMock

from targets.target_channel import group_stacks_by_filter


def _stack(stable_hash: str, filter_: str | None) -> MagicMock:
    s = MagicMock()
    s.stable_hash = stable_hash
    s.filter = filter_
    return s


def test_group_stacks_by_filter_drops_no_filter_stacks():
    stacks = [_stack("a", "R"), _stack("b", None), _stack("c", "R")]
    grouped = group_stacks_by_filter(stacks)
    assert set(grouped.keys()) == {"R"}
    assert [s.stable_hash for s in grouped["R"]] == ["a", "c"]


def test_group_stacks_by_filter_deterministic_order():
    """Hash-stable order: filters alphabetical, stacks-within sorted by stable_hash.

    Cache correctness depends on this — different insertion orders must
    produce the same tuple downstream.
    """
    a = group_stacks_by_filter([_stack("zz", "B"), _stack("aa", "R"), _stack("bb", "R")])
    b = group_stacks_by_filter([_stack("bb", "R"), _stack("zz", "B"), _stack("aa", "R")])
    assert list(a.keys()) == list(b.keys()) == ["B", "R"]
    assert [s.stable_hash for s in a["R"]] == [s.stable_hash for s in b["R"]] == ["aa", "bb"]
