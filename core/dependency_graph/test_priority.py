"""Pure — no Ray. A starred cart entry's sub-graph carries priority 1, its nested
sub-graphs inherit it, and the dispatch queue hands its nodes out before
unstarred work that is ready — without disturbing the depth-first order within
a priority."""
from types import SimpleNamespace

from dependency_graph.classes import Graph
from dependency_graph.dependency_graph import _queue
from dependency_graph.graph_runtime import GraphRuntime


def test_a_sub_graph_s_priority_is_its_own_or_inherited():
    root = GraphRuntime(Graph({}))
    starred = GraphRuntime(Graph({}, priority=1), parent=root)
    nested = GraphRuntime(Graph({}), parent=starred)
    plain = GraphRuntime(Graph({}), parent=root)
    assert (root.priority, starred.priority, nested.priority, plain.priority) == (0, 1, 1, 0)


def test_starred_work_is_popped_first_and_depth_first_order_stands_within_a_priority():
    hi, lo = SimpleNamespace(priority=1), SimpleNamespace(priority=0)
    queue = []
    _queue(queue, [(lo, "a1"), (lo, "a2")])          # an unstarred batch: a1 traced first
    _queue(queue, [(hi, "s1"), (lo, "b1"), (hi, "s2")])
    popped = [queue.pop()[1] for _ in range(len(queue))]
    assert popped[:2] == ["s1", "s2"]                # the starred nodes, first-traced first
    assert popped[2:] == ["b1", "a1", "a2"]          # then the newest unstarred batch, depth-first
