"""Unit tests for `paths_for_item` and its inner graph walker.

The inner `_enumerate_paths` is the pure-functional core (graph in,
paths out); we hand-build tiny graphs of synthetic Nodes to verify
topological resolution, the `finals_only` filter, and graceful
handling of missing `filename_fn`s. The outer `paths_for_item` does
context lookups + pipeline tracing which is covered by smoke tests
elsewhere.
"""
from __future__ import annotations

from pathlib import Path

from dependency_graph.classes import Graph, Node
from dependency_graph.trace_paths import _enumerate_paths


def _make_task(name, *, filename_fn=None, final=False, n_params=0):
    """Construct a synthetic task. `n_params` controls the placeholder
    `_function`'s parameter count so `_enumerate_paths`'s signature-
    based "pad args with ancestor results" rule sees the right
    arity (mirrors `graph_runtime.make_fn_args`)."""
    params = ", ".join(f"p{i}" for i in range(n_params))
    src = f"def f({params}): pass"
    ns: dict = {}
    exec(src, ns)
    f = ns["f"]
    f.__name__ = name
    f.__module__ = "test_trace_paths"

    # _function is set as an instance attribute (not class attribute)
    # so accessing it via the instance doesn't bind it as a method and
    # strip the first param from inspect.signature.
    class Task:
        pass
    t = Task()
    t._function = f
    if filename_fn is not None:
        t.filename_fn = filename_fn
    if final:
        t._final = True
    return t


def _node(name, task) -> Node:
    return Node(task, name=name)


def test_enumerate_paths_topological_order_resolves_downstream():
    """A → B chain: B's filename_fn sees A's resolved path as its
    one ancestor result. Both are returned in the path list."""
    a = _node("a", _make_task(
        "a", filename_fn=lambda: Path("/tmp/a.xisf"),
    ))
    # B's filename_fn receives the upstream Path as its sole arg.
    b = _node("b", _make_task(
        "b",
        filename_fn=lambda upstream: Path(f"{upstream}.b"),
        n_params=1,
    ))
    graph = Graph({a: [], b: [a]})

    paths = _enumerate_paths(graph, finals_only=False)
    assert paths == [Path("/tmp/a.xisf"), Path("/tmp/a.xisf.b")]


def test_enumerate_paths_finals_only_filters_to_marked_nodes():
    """`finals_only=True` returns only Nodes whose task has
    `_final == True`. The non-final upstream's path is still used
    internally to resolve the final downstream's args."""
    a = _node("a", _make_task(
        "a", filename_fn=lambda: Path("/tmp/a.xisf"), final=False,
    ))
    b = _node("b", _make_task(
        "b",
        filename_fn=lambda upstream: Path(f"{upstream}.b"),
        final=True,
        n_params=1,
    ))
    graph = Graph({a: [], b: [a]})

    paths = _enumerate_paths(graph, finals_only=True)
    assert paths == [Path("/tmp/a.xisf.b")]


def test_enumerate_paths_skips_nodes_without_filename_fn():
    """A Node whose task has no filename_fn is skipped entirely
    (not in the output, and its downstreams that depend on its
    "resolved path" are also skipped — there's nothing to feed
    forward)."""
    a = _node("a", _make_task("a"))  # no filename_fn
    b = _node("b", _make_task(
        "b",
        filename_fn=lambda upstream: Path(f"{upstream}.b"),
        n_params=1,
    ))
    graph = Graph({a: [], b: [a]})

    paths = _enumerate_paths(graph, finals_only=False)
    assert paths == []


def test_enumerate_paths_skips_downstream_when_upstream_fails():
    """If an upstream filename_fn raises, the downstream that
    depends on it is also skipped. Independent branches still
    resolve."""
    bad = _node("bad", _make_task(
        "bad", filename_fn=lambda: 1 / 0,
    ))
    downstream = _node("downstream", _make_task(
        "downstream",
        filename_fn=lambda upstream: Path(f"{upstream}.d"),
        n_params=1,
    ))
    independent = _node("independent", _make_task(
        "independent", filename_fn=lambda: Path("/tmp/ok.xisf"),
    ))
    graph = Graph({bad: [], downstream: [bad], independent: []})

    paths = _enumerate_paths(graph, finals_only=False)
    assert paths == [Path("/tmp/ok.xisf")]


def test_enumerate_paths_handles_diamond():
    """Diamond: root → (left, right) → sink. All four resolve;
    sink's filename_fn receives both left's and right's resolved
    paths in dep order."""
    root = _node("root", _make_task(
        "root", filename_fn=lambda: Path("/r"),
    ))
    left = _node("left", _make_task(
        "left",
        filename_fn=lambda upstream: Path(f"{upstream}/L"),
        n_params=1,
    ))
    right = _node("right", _make_task(
        "right",
        filename_fn=lambda upstream: Path(f"{upstream}/R"),
        n_params=1,
    ))
    sink = _node("sink", _make_task(
        "sink",
        filename_fn=lambda lp, rp: Path(f"{lp}+{rp}"),
        n_params=2,
    ))
    graph = Graph({
        root: [],
        left: [root],
        right: [root],
        sink: [left, right],
    })

    paths = _enumerate_paths(graph, finals_only=False)
    assert Path("/r") in paths
    assert Path("/r/L") in paths
    assert Path("/r/R") in paths
    assert Path("/r/L+/r/R") in paths
