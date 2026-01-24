import time
from functools import wraps
from typing import Callable
import ray

from protocols import CacheProtocol

_MAX_PERFORMANCE_SAMPLES = 100


def _update_performance(fn_name: str, elapsed_time: float, cache: CacheProtocol):
    node_ip = ray.util.get_node_ip_address()
    key = ("performance_history", node_ip, fn_name)
    cache.and_equals(key, [elapsed_time], _MAX_PERFORMANCE_SAMPLES)


def make_measure_performance_decorator(cache: CacheProtocol):
    def measure_performance(cache: CacheProtocol):
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                _update_performance(func.__name__, elapsed, cache)
                return result

            return wrapper

        return decorator

    return measure_performance(cache)
