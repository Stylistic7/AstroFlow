import ray
import time
from functools import wraps
from dataclasses import dataclass

from context import context


# After running hundreds of thousands of tasks, these are going to get
# unweildy.
@dataclass
class PerformanceSample:
  count: int
  time: float

  def __add__(self, time):
    self.time = (self.time * self.count + time) / (self.count + 1)
    self.count += 1

    return self


def track_performance(fn):

  @wraps(fn)
  def wrapper(*args, **kwargs):
    start_time = time.time()
    result = fn(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    node_ip = ray.util.get_node_ip_address()
    key = ('performance_history', node_ip, fn.__name__)
    context.and_equals(key, elapsed_time, default_value=PerformanceSample(0, 0))

    return result

  return wrapper
