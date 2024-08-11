import ray_local

import ray
import unittest
import time

from context import context
from throughput import track_performance, estimate_time


@ray.remote
@track_performance
def long_task(sleep_time):
  time.sleep(sleep_time)


class TestThroughput(unittest.TestCase):

  def test_track_performance(self):
    key = ('performance_history', ray.util.get_node_ip_address(), 'long_task')

    ray.get(long_task.remote(2))
    self.assertAlmostEqual(context[key].time, 2.0, 1)

    ray.get(long_task.remote(4))
    self.assertAlmostEqual(context[key].time, 3.0, 1)

    estimate = estimate_time('long_task', 2)
    self.assertAlmostEqual(estimate, 6.0, 1)


if __name__ == '__main__':
  unittest.main()
