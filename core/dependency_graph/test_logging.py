from ray_infra import ray_local

from typing import TypeVar, Any
import os
import unittest
import ray
import time
import structlog

from context import get_cache
from context import context
from dependency_graph import Graph, Node, GraphStatus, TopologicalSorterWithStatus, canonicalize_graph, run_graph, print_graph, make_graph_from_list
from dependency_graph.logging import capture_logs

cache = get_cache()
log = structlog.get_logger()

T = TypeVar('T')


@ray.remote
@capture_logs
def first_remote_task(arg: T) -> T:
  log.info(f'first_remote_task {arg}')
  return arg


@ray.remote
@capture_logs
def second_remote_task(arg: T) -> T:
  log.info(f'second_remote_task {arg}')
  return arg


class TestLogging(unittest.TestCase):

  def test_simple_graph(self) -> None:
    A = (first_remote_task, 'A')
    B = (first_remote_task, 'B')
    C = (second_remote_task, 'C')
    D = (second_remote_task, 'D')

    graph = Graph({D: {B, C}, C: {A}, B: {A}})
    results, logs = run_graph(graph)

    self.assertEqual(results[A], 'A')
    self.assertEqual(results[B], 'B')
    self.assertEqual(results[C], 'C')
    self.assertEqual(results[D], 'D')

    self.assertEqual(logs[A][0]['event'], 'first_remote_task A')
    self.assertEqual(logs[B][0]['event'], 'first_remote_task B')
    self.assertEqual(logs[C][0]['event'], 'second_remote_task C')
    self.assertEqual(logs[D][0]['event'], 'second_remote_task D')

    self.assertEqual(results.num_nodes_run, 4)
    self.assertEqual(results.max_parallelism_seen, 2)


if __name__ == '__main__':
  unittest.main()
