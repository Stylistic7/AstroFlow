from ray_infra import ray_local

import unittest
import ray
import time

from dependency_graph import Graph, run_graph, graph_status_to_progress
from decorators import cached


@cached.remote
def gather():
  pass


@cached.remote
def process(runtime, i):
  time.sleep(runtime)


def scatter(task, runtime, number):
  return Graph({gather: {(task, runtime, i) for i in range(number)}})


class TestProgress(unittest.TestCase):

  def test_one_process(self):
    counter = 10

    def status_fn(status):
      nonlocal counter

      progress = graph_status_to_progress(status)

      if counter == 0:
        return

      self.assertEqual(progress['process']['remaining_number'], counter)
      self.assertAlmostEqual(progress['process']['estimated_time'],
                             counter,
                             places=0)

      counter -= 1

    graph = scatter(process, runtime=1, number=10)
    run_graph(graph, status_fn=status_fn, status_interval=0)


if __name__ == '__main__':
  unittest.main()
