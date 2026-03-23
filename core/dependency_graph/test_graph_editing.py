from typing import TypeVar
import unittest
import ray

from dependency_graph import Graph, ResultDict, run_graph, print_graph
from dependency_graph.editing import replace_task
from dependency_graph.debugging import print_graph

T = TypeVar("T")


def _capturing_graph(graph: Graph) -> tuple[Graph, ResultDict]:
    """Attach a node_status_fn that captures results into a ResultDict."""
    captured = ResultDict()

    def on_node_done(node, result, logs):
        captured[node] = result

    graph.node_status_fn = on_node_done
    return graph, captured


@ray.remote
def first_task() -> int:
    return 1


@ray.remote
def later_task(i: int) -> int:
    return i + 1


@ray.remote
def last_task(i: int) -> int:
    return i + 3


@ray.remote
def fake_later_task(i: int) -> int:
    return i + 101


class TestGraphEditing(unittest.TestCase):
    def test_pass_down(self) -> None:
        A = first_task
        B = later_task
        C = last_task

        graph, captured = _capturing_graph(Graph({C: {B}, B: {A}}))
        stats = run_graph(graph)

        self.assertEqual(captured[A], 1)
        self.assertEqual(captured[B], 2)
        self.assertEqual(captured[C], 5)

        self.assertEqual(stats.num_nodes_run, 3)
        self.assertEqual(stats.max_parallelism_seen, 1)

        graph = replace_task(Graph({C: {B}, B: {A}}), later_task, fake_later_task)
        print_graph(graph)

        graph, captured = _capturing_graph(graph)
        stats = run_graph(graph)

        self.assertEqual(captured[A], 1)
        self.assertTrue(B not in captured)
        self.assertEqual(captured[fake_later_task], 102)
        self.assertEqual(captured[C], 105)

        self.assertEqual(stats.num_nodes_run, 3)
        self.assertEqual(stats.max_parallelism_seen, 1)
