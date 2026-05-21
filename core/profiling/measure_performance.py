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

from hashing import hash_source_code
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


def _update_profile(
    fn_hash: str,
    fn_name: str,
    quality: str | None,
    record: dict[str, Any],
    cache: CacheProtocol,
):
    node_ip = ray.util.get_node_ip_address()
    record["node_ip"] = node_ip
    key = ("profile", fn_hash, fn_name, quality)
    cache.and_equals(key, [record], _MAX_PROFILE_SAMPLES)


def _infer_quality(args, kwargs) -> str | None:
    """Scan args/kwargs for a quality-level string; return it or None.

    Quality is a free-form string now (e.g. "Master", "Subframe"); the
    legacy QualityLevel enum is gone. We match against the known
    values in ALL_QUALITY_LEVELS so a random string arg can't be
    misread as a quality. Also looks one level inside dataclasses
    (legacy ProcessingConfig.quality is gone, but this still gives
    us robustness for any user dataclass that carries a quality).
    """
    from previews.types import ALL_QUALITY_LEVELS
    known = set(ALL_QUALITY_LEVELS)
    for v in (*args, *kwargs.values()):
        if isinstance(v, str) and v in known:
            return v
        if is_dataclass(v) and not isinstance(v, type):
            for f in fields(v):
                inner = getattr(v, f.name, None)
                if isinstance(inner, str) and inner in known:
                    return inner
    return None


def make_measure_performance_decorator(cache: CacheProtocol):
    def decorator(func: Callable, ray_opts: dict[str, Any] | None = None):
        ray_opts = ray_opts or {}
        fn_hash = hash_source_code(func)

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
                # ray_opts can carry callables (e.g. cache_key_narrow);
                # a profile record must be pure data — it's persisted
                # and JSON-served at /api/performance.
                "ray_opts": {
                    k: v for k, v in ray_opts.items() if not callable(v)
                },
            }

            quality = _infer_quality(args, kwargs)
            _update_profile(fn_hash, func.__name__, quality, record, cache)
            return result

        return wrapper

    return decorator
