"""Pure — no Ray. The periodic workflow-graph render fires only for a runtime
whose state changed since its last render — a change anywhere below it counts
— and a concluded sub-graph renders once more, then never again (2026-11-06:
re-rendering every pipeline every tick kept the dispatch loop from
dispatching)."""
from dependency_graph.classes import Graph, Node
from dependency_graph.dependency_graph import _fire_graph_updates
from dependency_graph.graph_runtime import GraphRuntime


class _FakeRemote:
    def __init__(self, name):
        self._function = type(name, (), {"__module__": "test", "__name__": name})()


def _node(name):
    return Node(_FakeRemote(name))


def test_only_a_changed_pipeline_renders_and_a_concluded_one_renders_once_more():
    renders = []
    P, X, Y = _node("P"), _node("X"), _node("Y")
    root = GraphRuntime(Graph({P: []}))
    root.start(P)
    root.node_completed(P, Graph({X: [], Y: [X]}, graph_update_fn=lambda rt: renders.append(rt)))
    pipeline = root._children[P]

    assert _fire_graph_updates(root) == 1           # first sight of it
    assert _fire_graph_updates(root) == 0           # nothing happened since
    pipeline.start(X)
    assert _fire_graph_updates(root) == 1           # a node started
    pipeline.node_completed(X, "x")
    pipeline.start(Y)
    pipeline.node_completed(Y, "y")
    assert not root.is_active()                     # the pipeline concluded
    assert _fire_graph_updates(root) == 1           # its final state
    assert _fire_graph_updates(root) == 0           # and never again
    assert renders == [pipeline] * 3
