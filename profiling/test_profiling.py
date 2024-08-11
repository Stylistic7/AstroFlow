import ray_local

import unittest
import ray
import time

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

    if context.contains(key):
      print(f'### CONTAINS KEY {context[key]}')

    self.assertFalse(context.contains(key))

    ray.get(long_task.remote(2))
    self.assertAlmostEqual(context[key].time, 2.0, 1)

    ray.get(long_task.remote(4))
    self.assertAlmostEqual(context[key].time, 3.0, 1)

    estimate = estimate_time('long_task', 2)
    self.assertAlmostEqual(estimate, 6.0, 1)

  def test_missing_history(self):
    estimate = estimate_time('never_executed_task')
    self.assertEqual(estimate, None)

  def test_dependency_graph(self):
    graph = {scatter: {}}

    final_status = None

    def function_name_from_node(node):
      return node.task._function.__name__

    def print_status(status):
      print('### status', status)

      total_estimated_time = 0
      for node in status.ready_nodes | status.waiting_nodes:
        if estimated_time := estimate_time(function_name_from_node(node)):
          total_estimated_time += estimated_time

      print('### total_estimated_time', total_estimated_time)
      nonlocal final_status
      final_status = status

    results = run_graph(graph, status_fn=print_status)

    self.assertEqual(results[scatter], 21)

    self.assertEqual(results.num_nodes_run, 9)
    self.assertEqual(results.max_parallelism_seen, 7)

    self.assertEqual(len(final_status.done_nodes), 9)
    self.assertEqual(len(final_status.ready_nodes), 0)
    self.assertEqual(len(final_status.waiting_nodes), 0)

    estimated_time = estimate_time('process')
    print('### estimated time for process() function is', estimated_time)
    self.assertIsNotNone(estimated_time)


if __name__ == '__main__':
  unittest.main()
