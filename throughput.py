import ray
import os
import time
from functools import wraps
from dataclasses import dataclass
import ipaddress

from cache import cache
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


def estimate_time(tag, number=1):
  node_ips = []
  resources = ray.cluster_resources()
  for k, v in resources.items():
    try:
      _, ip = k.split('node:')
      ip = ipaddress.ip_address(ip)
      node_ips.append(str(ip))
    except:
      pass

  total_count = 0
  total_time = 0
  for node_ip in node_ips:
    key = ('performance_history', node_ip, tag)
    history = context.get(key, None)
    if history is None:
      continue

    total_time += history.time * history.count
    total_count += history.count

  if total_count:
    return number * (total_time / total_count)


if __name__ == '__main__':
  print('foo')
