import ray
import time
from functools import wraps
from dataclasses import dataclass
from typing import ParamSpec, Callable, TypeVar

from context import context

_MAX_PERFORMANCE_SAMPLES = 100

P = ParamSpec("P")
R = TypeVar("R")


# TODO: Nodes need to be identified by something more stable and reliable than
# IP address. Many users' IP addresses change daily or even hourly. Eventually,
# some kind of authentication system will keep track of users. Maybe MAC
# address is a good choice.
def track_performance(fn: Callable[P, R]) -> Callable[P, R]:

  @wraps(fn)
  def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
    start_time = time.time()
    result = fn(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time

    node_ip = ray.util.get_node_ip_address()

    print(f'### Function {fn.__name__} took {elapsed_time:.3f} s on {node_ip}')
    key = ('performance_history', node_ip, fn.__name__)

    # Keep only the last N samples.
    history = context[key] or []
    history = history[-(_MAX_PERFORMANCE_SAMPLES - 1):] + [elapsed_time]
    context[key] = history

    return result

  return wrapper
