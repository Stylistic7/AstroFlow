import ray_local

import ray
import unittest
import time

from context import context
import throughput
from throughput import track_performance


@ray.remote
@track_performance
def long_task(sleep_time):
  time.sleep(sleep_time)


class TestThroughput(unittest.TestCase):

  def test_one_value(self):
    count, mean = 0, 0
    count, mean = throughput._add_sample(count, mean, 10)

    self.assertEqual(count, 1)
    self.assertEqual(mean, 10)

  def test_two_values(self):
    count, mean = 0, 0
    count, mean = throughput._add_sample(count, mean, 10)
    count, mean = throughput._add_sample(count, mean, 20)

    self.assertEqual(count, 2)
    self.assertEqual(mean, 15)

  def track_performance(self):
    key = ('performance_history', ray.util.get_node_ip_address(), 'long_task')

    ray.get(long_task.remote(2))
    self.assertAlmostEqual(context[key].time, 2.0)

    ray.get(long_task.remote(4))
    self.assertAlmostEqual(context[key].time, 3.0)


if __name__ == '__main__':
  unittest.main()
