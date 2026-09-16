"""Readiness is an event: `drain_newly_ready` hands the dispatcher exactly
the nodes that became ready since the last drain — the initial frontier in
graph order, each dependent as its deps complete, a dynamic child's
frontier once it is spliced in — and never a node twice, never one that
was started or completed meanwhile (2026-09-15)."""
from dependency_graph.classes import Graph, Node
from dependency_graph.graph_runtime import GraphRuntime


class _FakeRemote:
    def __init__(self, name):
        self._function = type(name, (), {"__module__": "test", "__name__": name})()


def _node(name):
    return Node(_FakeRemote(name))


def _names(pairs):
    return [n.task._function.__name__ for _, n in pairs]


def test_the_initial_frontier_drains_once_in_graph_order():
    A, B, C, D = (_node(x) for x in "ABCD")
    rt = GraphRuntime(Graph({A: [], B: [], C: [A, B], D: []}))
    assert _names(rt.drain_newly_ready()) == ["A", "B", "D"]
    assert rt.drain_newly_ready() == []


def test_a_completion_surfaces_only_the_dependents_it_unblocked():
    A, B, C = (_node(x) for x in "ABC")
    rt = GraphRuntime(Graph({C: [A, B]}))
    rt.drain_newly_ready()
    rt.start(A); rt.node_completed(A, 1)
    assert rt.drain_newly_ready() == []          # C still waits on B
    rt.start(B); rt.node_completed(B, 2)
    assert _names(rt.drain_newly_ready()) == ["C"]
    assert rt.drain_newly_ready() == []


def test_nodes_started_or_completed_before_the_drain_are_dropped():
    A, B = _node("A"), _node("B")
    rt = GraphRuntime(Graph({B: [A]}))
    rt.start(A)                                  # started before anyone drained
    rt.node_completed(A, 1)                      # ...and completed (a driver-side hit)
    rt.start(B); rt.node_completed(B, 2)
    assert rt.drain_newly_ready() == []
    assert not rt.is_active()


def test_a_dynamic_child_contributes_its_frontier_tagged_with_itself():
    A, B = _node("A"), _node("B")
    inner = _node("inner")
    rt = GraphRuntime(Graph({B: [A]}))
    rt.drain_newly_ready()
    rt.start(A); rt.node_completed(A, Graph({inner: []}))
    pairs = rt.drain_newly_ready()
    assert _names(pairs) == ["inner"]
    (child, _), = pairs
    assert child is not rt and child.parent is rt
    child.start(inner); child.node_completed(inner, "x")
    assert _names(rt.drain_newly_ready()) == ["B"]  # the child concluded, A's result landed
    assert rt.started(A) and not rt.started(B)
