from typing import TypeVar
import unittest
import ray

from dependency_graph import Graph, run_graph, print_graph
from dependency_graph.editing import replace_task
from dependency_graph.debugging import print_graph

T = TypeVar("T")


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

        graph = Graph({C: {B}, B: {A}})
        results, _ = run_graph(graph)

        self.assertEqual(results[A], 1)
        self.assertEqual(results[B], 2)
        self.assertEqual(results[C], 5)

        self.assertEqual(results.num_nodes_run, 3)
        self.assertEqual(results.max_parallelism_seen, 1)

        graph = replace_task(graph, later_task, fake_later_task)
        print_graph(graph)

        results, _ = run_graph(graph)

        self.assertEqual(results[A], 1)
        self.assertTrue(B not in results)
        self.assertEqual(results[fake_later_task], 102)
        self.assertEqual(results[C], 105)

        self.assertEqual(results.num_nodes_run, 3)
        self.assertEqual(results.max_parallelism_seen, 1)
