import ray
import os
import time
from functools import wraps
from dataclasses import dataclass

from cache import cache
from context import context


# After running hundreds of thousands of tasks, these are going to get
# unweildy.
@dataclass
class PerformanceSample:
  count: int
  time: float

  def __add__(self, time):
    print(f'#### in __add__')
    self.time = (self.time * self.count + time) / (self.count + 1)
    self.count += 1

    return self


def track_performance(fn):

  @wraps(fn)
  def wrapper(*args, **kwargs):
    start_time = time.time()
    fn(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    node_ip = ray.util.get_node_ip_address()
    key = ('performance_history', node_ip, fn.__name__)
    context.and_equals(key, elapsed_time, default_value=PerformanceSample(0, 0))

  return wrapper


# TODO: This needs to upgraded to be "Ray-aware." Tasks run at different
# speeds on different nodes. How can we account for that in our estimates?
# We also need to get the Ray worker's IP or hostname to track this.
def _add_sample(prev_count, prev_mean, time):
  return (prev_count + 1, (prev_mean * prev_count + time) / (prev_count + 1))


def add_sample(tag, time):
  key = ('throughput_history', tag)
  with cache.transact():
    prev_count, prev_mean = cache.get(key, (0, 0))
    cache[key] = _add_sample(prev_count, prev_mean, time)


# TODO: This needs to be upgraded to be "Ray-aware". Ray can tell us how many
# workers there, on which hosts.
def estimate_time(tag, number=1, parallelism=None):
  if not parallelism:
    parallelism = os.cpu_count()

  key = ('throughput_history', tag)
  _, mean = cache.get(key, (0, 0))

  return mean * number / parallelism


if __name__ == '__main__':
  print('foo')
