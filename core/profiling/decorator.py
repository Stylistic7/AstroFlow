import ray
import time
from functools import wraps
from dataclasses import dataclass

from context import context

_MAX_PERFORMANCE_SAMPLES = 20


# TODO: Nodes need to be identified by something more stable and reliable than
# IP address. Many users' IP addresses change daily or even hourly. Eventually,
# some kind of authentication system will keep track of users. Maybe MAC
# address is a good choice.
def track_performance(fn):

  @wraps(fn)
  def wrapper(*args, **kwargs):
    start_time = time.time()
    result = fn(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    node_ip = ray.util.get_node_ip_address()
    key = ('performance_history', node_ip, fn.__name__)

    # Keep only the last N samples.
    history = context[key] or []
    history = history[-_MAX_PERFORMANCE_SAMPLES - 1:]
    history.append(elapsed_time)
    context[key] = history

    return result

  return wrapper
