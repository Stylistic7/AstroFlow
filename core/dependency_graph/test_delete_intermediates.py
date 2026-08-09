"""`Graph(..., delete_intermediates=True)` — a dynamic subgraph that cleans
up after itself, and only after itself.

Two properties, both end-to-end through the real runtime:

1. When the subgraph concludes successfully, every `@makes_image` node in it
   EXCEPT the single terminal has its output deleted and its cache entry
   invalidated. The terminal escapes and persists.
2. On a later run the terminal is a cache HIT, so the whole subgraph is
   skipped at splice time — the intermediates are never re-derived just to be
   deleted again.

Plus unit-level coverage of the failure rule (a failed subgraph keeps its
files, so a retry resumes from real data).
"""
from pathlib import Path

import pytest

from context import context
from dependency_graph import run_graph
from dependency_graph.classes import Graph, Node
from dependency_graph.graph_runtime import GraphRuntime
from dependency_graph._delete_intermediates_fixtures import (
    build_top_level_graph, executions,
)


def _fits(out_dir: Path, op_dir: str) -> list:
    return list((Path(out_dir) / op_dir).glob("*.fits"))


def test_intermediates_deleted_and_subgraph_skipped_next_run(test_context):
    # macOS's temp dir is a symlink (/var -> /private/var) and
    # `compute_image_metadata` stores the RESOLVED filename, so resolve here
    # or the produced paths won't match the output dir the test globs.
    context.output_dir = Path(context.output_dir).resolve()
    out = Path(context.output_dir)
    seed, config = 7, "cfg-v1"

    # ---- RUN 1: everything runs; intermediates die at conclusion --------
    run_graph(build_top_level_graph(seed, config))

    assert executions(out, "_di_source") == 1
    assert executions(out, "_di_middle") == 1
    assert executions(out, "_di_terminal") == 1

    assert len(_fits(out, "_di_terminal")) == 1, (
        "the terminal escapes the subgraph and must persist"
    )
    # BOTH intermediates go — not just the terminal's immediate predecessor.
    assert _fits(out, "_di_middle") == [], "middle intermediate not deleted"
    assert _fits(out, "_di_source") == [], "source intermediate not deleted"

    # ---- RUN 2: terminal is cached → the subgraph never expands ---------
    run_graph(build_top_level_graph(seed, config))

    assert executions(out, "_di_source") == 1, (
        "the subgraph expanded again and re-derived its intermediates; the "
        "terminal was already a cache hit, so it should have been skipped"
    )
    assert executions(out, "_di_middle") == 1
    assert executions(out, "_di_terminal") == 1
    assert len(_fits(out, "_di_terminal")) == 1
    assert _fits(out, "_di_middle") == []


# ── unit level: the rules, without Ray ────────────────────────────────────

class _FakeRemote:
    """Minimal task stand-in: Node needs `task._function`."""
    def __init__(self, name):
        def fn():
            pass
        fn.__name__ = name
        fn.__module__ = "test_delete_intermediates"
        self._function = fn


def _node(name):
    return Node(_FakeRemote(name), name=name)


def _spliced(monkeypatch, child_graph):
    """Splice `child_graph` in as a dynamic subgraph and return
    (root, parent_node, child)."""
    monkeypatch.setattr(
        "dependency_graph.dependency_graph._apply_preflight",
        lambda g, is_subgraph=False: g,
    )
    parent_node = _node("parent")
    rt = GraphRuntime(Graph({parent_node: []}))
    rt.start(parent_node)
    rt.node_completed(parent_node, child_graph)
    return rt, parent_node, rt._children[parent_node]


def test_failed_subgraph_keeps_its_intermediates(monkeypatch):
    """A failed scope must NOT delete: the retry resumes from real files
    instead of re-deriving the whole subgraph."""
    calls = []
    monkeypatch.setattr(
        "cache.reclaim.reclaim_artifact",
        lambda node, result, fn_args: calls.append(node),
    )
    interior, terminal = _node("interior"), _node("terminal")
    _, _, child = _spliced(
        monkeypatch,
        Graph({terminal: [interior]}, delete_intermediates=True),
    )

    child.node_completed(interior, "interior-result")
    child.node_failed(terminal, RuntimeError("boom"))
    # Draining the parent is what fires conclusion handling.
    child.parent._propagate_completed_children()

    assert calls == [], "a failed subgraph must keep its intermediates"


def test_plain_subgraph_deletes_nothing(monkeypatch):
    """Without the flag, a concluded subgraph keeps everything — the default
    is the pure world: every image persists."""
    calls = []
    monkeypatch.setattr(
        "cache.reclaim.reclaim_artifact",
        lambda node, result, fn_args: calls.append(node),
    )
    interior, terminal = _node("interior"), _node("terminal")
    _, _, child = _spliced(monkeypatch, Graph({terminal: [interior]}))

    child.node_completed(interior, "interior-result")
    child.node_completed(terminal, "terminal-result")
    child.parent._propagate_completed_children()

    assert calls == []
