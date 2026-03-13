import os
import time
import threading
from functools import wraps
from typing import Callable, Any
import inspect
import psutil
import ray
from collections.abc import Mapping
from dataclasses import is_dataclass, fields

from protocols import CacheProtocol
from metadata.classes import ImageMetadata

_MAX_PROFILE_SAMPLES = 100
_MEMORY_SAMPLE_INTERVAL = 0.002


def _collect_image_stats(obj):

    count = 0
    pixels = 0
    bytes_ = 0

    seen = set()
    stack = [obj]

    while stack:
        v = stack.pop()

        obj_id = id(v)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        if isinstance(v, ImageMetadata):
            count += 1
            pixels += v.pixels
            bytes_ += v.total_bytes
            continue

        if isinstance(v, (list, tuple, set, frozenset)):
            stack.extend(v)
            continue

        if isinstance(v, Mapping):
            stack.extend(v.values())
            continue

        if is_dataclass(v) and not isinstance(v, type):
            stack.extend(getattr(v, f.name) for f in fields(v))
            continue

        d = getattr(v, "__dict__", None)
        if d:
            stack.extend(d.values())

    return {
        "count": count,
        "pixels": pixels,
        "bytes": bytes_,
    }


def _collect_image_stats_by_argument(func, args, kwargs):
    bound = inspect.signature(func).bind_partial(*args, **kwargs)
    bound.apply_defaults()

    by_arg = {
        name: stats
        for name, value in bound.arguments.items()
        if (stats := _collect_image_stats(value))["count"] > 0
    }

    totals = {
        "count": sum(s["count"] for s in by_arg.values()),
        "pixels": sum(s["pixels"] for s in by_arg.values()),
        "bytes": sum(s["bytes"] for s in by_arg.values()),
    }

    return by_arg, totals


def _update_profile(fn_name: str, record: dict[str, Any], cache: CacheProtocol):
    node_ip = ray.util.get_node_ip_address()
    key = ("profile", node_ip, fn_name)
    cache.and_equals(key, [record], _MAX_PROFILE_SAMPLES)


def make_measure_performance_decorator(cache: CacheProtocol):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process(os.getpid())
            baseline_rss = process.memory_info().rss
            peak_rss = baseline_rss
            stop_event = threading.Event()

            def sampler():
                nonlocal peak_rss
                while not stop_event.is_set():
                    peak_rss = max(peak_rss, process.memory_info().rss)
                    stop_event.wait(_MEMORY_SAMPLE_INTERVAL)

            image_stats_by_arg, image_totals = _collect_image_stats_by_argument(
                func, args, kwargs
            )

            t = threading.Thread(target=sampler, daemon=True)
            wall_start = time.perf_counter()
            cpu_start = time.process_time()
            t.start()

            try:
                result = func(*args, **kwargs)
            finally:
                stop_event.set()
                t.join()

            record = {
                "wall_time_s": time.perf_counter() - wall_start,
                "cpu_time_s": time.process_time() - cpu_start,
                "peak_rss_delta_b": max(peak_rss, process.memory_info().rss)
                - baseline_rss,
                "images": image_stats_by_arg,
                "image_totals": image_totals,
            }

            _update_profile(func.__name__, record, cache)
            return result

        return wrapper

    return decorator
