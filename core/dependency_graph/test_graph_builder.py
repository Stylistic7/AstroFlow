"""Tests for `@graph_builder` + `set_workflow_key`.

These exercise the trace-recording side of the decorator (where it acts
as an `record(...)` shorthand for ops nested inside another active
trace). The Ray-dispatched `.remote(...)` side is exercised end-to-end
by the larger pipeline tests; here we only structurally assert that
`.remote` is wired up and accepts the runtime's `dry_run=` kwarg shape.
"""
import contextvars

import pytest

from dependency_graph.classes import Graph, Node
from dependency_graph.tracing import (
    _TRACE,
    graph_builder,
    record,
    set_workflow_key,
    trace,
)


def _reset_trace(token):
    _TRACE.reset(token)


def test_graph_builder_records_when_called_inside_trace():
    @graph_builder
    def inner(x):
        return x

    g = Graph()
    token = _TRACE.set(g)
    try:
        node = inner("hello")
    finally:
        _reset_trace(token)

    assert isinstance(node, Node)
    assert node in g
    # Literal arg captured on the node descriptor.
    assert node.args == ("hello",)


def test_graph_builder_raises_when_called_outside_trace():
    @graph_builder
    def inner(x):
        return x

    with pytest.raises(RuntimeError, match="outside a trace"):
        inner("hello")


def test_graph_builder_rejects_kwargs_in_trace():
    @graph_builder
    def inner(x, y=None):
        return (x, y)

    g = Graph()
    token = _TRACE.set(g)
    try:
        with pytest.raises(NotImplementedError, match="kwargs"):
            inner("hello", y="world")
    finally:
        _reset_trace(token)


def test_graph_builder_with_ray_opts_decorator():
    """`@graph_builder(num_cpus=…)` form returns a working decorator."""
    @graph_builder(num_cpus=0.1)
    def inner(x):
        return x

    g = Graph()
    token = _TRACE.set(g)
    try:
        node = inner(42)
    finally:
        _reset_trace(token)

    assert isinstance(node, Node)
    # Surface the runtime needs: ._function (sig inspection) and .remote.
    assert hasattr(inner, "_function")
    assert hasattr(inner, "remote")


def test_graph_builder_positional_only_form_rejects_positional_args():
    """`@graph_builder("bad", num_cpus=0.1)` is a typo — fail loudly."""
    with pytest.raises(TypeError, match="keyword arguments"):
        graph_builder("bad", num_cpus=0.1)


def test_graph_builder_traces_dependencies_correctly():
    """Handle args become dependency edges; literals stay as Node.args."""
    @graph_builder
    def step(x):
        return x

    g = Graph()
    token = _TRACE.set(g)
    try:
        a = step("a")
        b = step("b")
        # Mix of literals and a Node handle:
        c = step(a)
        # Two-handle dependency: c depends on both a and b literally.
        d = step(a, b, "literal")
    finally:
        _reset_trace(token)

    # a, b have no deps (only literals).
    assert g[a] == []
    assert g[b] == []
    # c depends on a (handle), no literal args.
    assert g[c] == [a]
    assert c.args == ()
    # d depends on a and b, with one literal.
    assert g[d] == [a, b]
    assert d.args == ("literal",)


def test_set_workflow_key_attaches_callbacks_to_active_trace():
    @graph_builder
    def step(x):
        return x

    def builder():
        set_workflow_key("test", "key", "parts")
        return step("x")

    g = trace(builder)
    assert g is not None
    assert g.graph_update_fn is not None
    assert g.node_status_fn is not None
    # The graph_update_fn is `partial(store_workflow_graph, key_tuple)`;
    # we don't want to import store_workflow_graph here just to compare
    # identities, so check the partial's first arg is the key tuple.
    assert g.graph_update_fn.args == (("test", "key", "parts"),)


def test_set_workflow_key_raises_outside_trace():
    with pytest.raises(RuntimeError, match="outside a trace"):
        set_workflow_key("x")


def test_graph_builder_none_return_propagates_via_trace():
    """A traced builder returning None means 'no graph for this input';
    `trace()` propagates that as None."""
    @graph_builder
    def step(x):
        return x

    def builder_that_bails():
        # Don't even record anything — just bail.
        return None

    g = trace(builder_that_bails)
    assert g is None
