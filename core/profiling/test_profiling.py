from ray_infra import ray_local

import unittest
import ray
import time
import numpy as np

from context import context
from dependency_graph import Graph, run_graph
from profiling.estimate_time import estimate_time
from profiling.decorator import track_performance


@ray.remote
def scatter():
  return Graph({gather: [(process, i) for i in range(7)]})


@ray.remote
@track_performance
def process(item):
  time.sleep(2)
  return item


@ray.remote
def gather(*, ancestor_results):
  return sum(ancestor_results)


@ray.remote
@track_performance
def long_task(sleep_time):
  time.sleep(sleep_time)


class TestProfiling(unittest.TestCase):

  def test_track_performance(self):
    key = ('performance_history', ray.util.get_node_ip_address(), 'long_task')
    self.assertFalse(context.contains(key))

    ray.get(long_task.remote(2))
    self.assertTrue(np.allclose(context[key], [2.0], atol=1e-1))

    ray.get(long_task.remote(4))
    self.assertTrue(np.allclose(context[key], [2.0, 4.0], atol=1e-1))

    estimate = estimate_time('long_task', 2)
    self.assertAlmostEqual(estimate, 6.0, 1)

  def test_missing_history(self):
    estimate = estimate_time('never_executed_task')
    self.assertEqual(estimate, None)

  def test_dependency_graph(self):
    graph = Graph({scatter: {}})

    def function_name_from_node(node):
      return node.task._function.__name__

    num_status_calls = 0
    last_status = None
    estimated_times = []

    def print_status(status):
      nonlocal num_status_calls, last_status, estimated_times
      num_status_calls += 1
      last_status = status

      total_estimated_time = 0
      for node in status.ready_nodes | status.waiting_nodes:
        if estimated_time := estimate_time(function_name_from_node(node)):
          total_estimated_time += estimated_time

      estimated_times.append(total_estimated_time)

    results = run_graph(graph, status_fn=print_status, status_interval=0)

    self.assertEqual(results[scatter], 21)

    self.assertEqual(results.num_nodes_run, 9)
    self.assertEqual(results.max_parallelism_seen, 7)

    self.assertEqual(len(last_status.done_nodes), 9)
    self.assertEqual(len(last_status.ready_nodes), 0)
    self.assertEqual(len(last_status.waiting_nodes), 0)

    self.assertGreaterEqual(num_status_calls, 10)
    self.assertEqual(len(last_status.done_nodes), 9)
    self.assertEqual(len(last_status.ready_nodes), 0)
    self.assertEqual(len(last_status.waiting_nodes), 0)

    estimated_time = estimate_time('process')
    self.assertIsNotNone(estimated_time)
    self.assertAlmostEqual(estimated_time, 2.0, 1)

    self.assertTrue(
        np.allclose(estimated_times, [0, 14, 12, 10, 8, 6, 4, 2, 0, 0],
                    atol=1e-1))


if __name__ == '__main__':
  unittest.main()
