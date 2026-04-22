import ray
import inspect
from functools import wraps
from typing import Any, Callable, Optional, get_origin, TypeVar, Generic
from pathlib import Path
import time
import objsize
from beartype import beartype as _beartype
from dataclasses import dataclass
from ray.runtime_context import get_runtime_context

from hashing import hash_source_code
from cache.context_ref import ContextRef
from protocols import CacheProtocol
from cache.types import NeverRun


from cache.context_ref import strip_refs

import os
TRACE_TASKS = os.environ.get("ASTROFLOW_TRACE_TASKS", "") == "1"


T = TypeVar("T")


@dataclass(frozen=True)
class Injected(Generic[T]):
    value: T


class MissingInjectedValueError(Exception):
    pass


def _is_head_node() -> bool:
    node_id = get_runtime_context().get_node_id()
    for n in ray.nodes():
        if n["Alive"] and n["NodeID"] == node_id:
            # Ray marks the head with this internal resource
            return "node:__internal_head__" in n["Resources"]
    raise RuntimeError("Current node not found in ray.nodes()")


def _bind_args(
    func: Callable, args: tuple, kwargs: dict, injected_context: Optional[dict]
) -> inspect.BoundArguments:
    sig = inspect.signature(func)
    bound = sig.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    for name, param in sig.parameters.items():
        if (
            get_origin(param.annotation) == Injected
            and name not in bound.arguments
        ):
            if injected_context is None or name not in injected_context:
                raise MissingInjectedValueError(
                    f"Missing injected argument: {name}"
                )
            bound.arguments[name] = Injected(value=injected_context[name])
    return bound


# TODO: Consider removing `injected_context` if it's not used anywhere. The
# original intent was to handle context for subgraphs, like temporary
# directories.
def make_cached_decorator(
    cache_provider: Callable[[], CacheProtocol],
    call_hasher: Optional[Callable[[str, Any], str]],
    memory_estimator: Optional[Callable[[Callable, tuple, dict], int]] = None,
    debug=False,
):
    cache: CacheProtocol | None = None

    def _resolve_cache():
        nonlocal cache
        if cache is None:
            cache = cache_provider()
        return cache

    def debug_print(*args, **kwargs):
        if debug:
            print(*args, **kwargs)

    def decorate(func: Callable, ray_opts: dict) -> Callable:
        fn_hash = hash_source_code(func)
        func = _beartype(func)

        def _get_cache_key_and_result(bound):
            call_hash = call_hasher(fn_hash, *bound.args, **bound.kwargs)
            key = (func.__name__, call_hash)

            if key not in _resolve_cache():
                debug_print(f"cache miss: {key}")
                return key, call_hash, False, None

            start_time = time.time()
            cached = strip_refs(_resolve_cache()[key])
            elapsed_time = time.time() - start_time
            if elapsed_time > 0.5:
                debug_print(
                    f"### cache key {key} was slow to load, took {elapsed_time=} s"
                )
            size_mb = objsize.get_deep_size(cached) / 1024.0 / 1024.0
            if size_mb > 5:
                debug_print(
                    f"### cache key {key} contained large object, {size_mb:0.3f} MB"
                )

            debug_print(f"cache hit: {key}")
            return key, call_hash, True, cached

        def inner(
            *args,
            injected_context: Optional[dict] = None,
            dry_run: bool = False,
            **kwargs,
        ):
            if debug:
                assert _is_head_node()

            bound = _bind_args(func, args, kwargs, injected_context)
            key, call_hash, valid, cached_result = _get_cache_key_and_result(
                bound
            )

            if valid:
                if isinstance(cached_result, ContextRef):
                    assert cached_result.ref is None
                return cached_result

            if dry_run:
                return NeverRun

            if TRACE_TASKS:
                print(f">>> {func.__name__} (cache miss, running)")
            start_time = time.time()
            result = func(*bound.args, **bound.kwargs)
            elapsed_time = time.time() - start_time
            if TRACE_TASKS:
                print(f"<<< {func.__name__} ({elapsed_time:.3f}s)")

            if elapsed_time > 0.5:
                debug_print(
                    f"### fn {func.__name__} was slow to run, took {elapsed_time=} s"
                )
            size_mb = objsize.get_deep_size(result) / 1024.0 / 1024.0
            if size_mb > 5:
                debug_print(
                    f"### fn {func.__name__} returned large object, {size_mb:0.3f} MB"
                )
            if isinstance(result, Path) and not result.exists():
                raise FileNotFoundError(result)

            debug_print(
                f"storing: {key} as {type(result)} into cache {_resolve_cache().directory}"
            )
            _resolve_cache()[key] = strip_refs(result)

            return result

        inner.__name__ = func.__name__
        inner.__qualname__ = func.__qualname__
        inner.__module__ = func.__module__

        # Pull our own kwargs out of ray_opts before handing to ray.remote
        # (which would reject unknown kwargs).
        auto_memory = ray_opts.pop("auto_memory", False)

        ray_opts.setdefault("scheduling_strategy", "SPREAD")

        ray_func = ray.remote(**ray_opts)(inner)

        @wraps(func)
        def wrapper(*args, **kwargs):
            raise RuntimeError(
                f"{func.__name__} is a remote task. Use `.remote(...)`."
            )

        if auto_memory:
            if memory_estimator is None:
                raise RuntimeError(
                    f"{func.__name__} uses auto_memory=True but this "
                    f"make_cached_decorator was built without a memory_estimator."
                )

            def _auto_memory_remote(*args, **kwargs):
                estimate = memory_estimator(func, args, kwargs)
                print(
                    f"### auto_memory: {func.__name__} → {estimate / 1024**3:.0f} GB"
                )
                return ray_func.options(memory=estimate).remote(*args, **kwargs)

            wrapper.remote = _auto_memory_remote
        else:
            wrapper.remote = ray_func.remote
        wrapper.options = ray_func.options
        wrapper._original = func
        wrapper._function = func
        wrapper._is_task = True

        # Forward all custom attributes from inner decorators (e.g.
        # filename_fn, thumbnail_fn, tile_pyramid_fn, _validity_entries)
        # so they survive the cached.remote wrapping.
        wrapper.__dict__.update(func.__dict__)

        # Update registry entries to point at the @node-wrapped function
        # (the one with .remote and _function) instead of the inner wrapper.
        if hasattr(wrapper, '_registry_keys'):
            from products.registry import _registry
            for media_type, obj_type, level, impl in wrapper._registry_keys:
                key = (obj_type, level, impl)
                if key in _registry[media_type]:
                    _registry[media_type][key] = wrapper

        def is_cached(
            *args, injected_context: Optional[dict] = None, **kwargs
        ) -> bool:
            bound = _bind_args(func, args, kwargs, injected_context)
            _, _, valid, _ = _get_cache_key_and_result(bound)
            return valid

        wrapper.is_cached = is_cached

        return wrapper

    class CachedTask:
        def __call__(self, func: Callable) -> Callable:
            return decorate(func, ray_opts={})

        def remote(self, func: Callable = None, **ray_opts):
            if func is None:
                return lambda f: decorate(f, ray_opts)
            return decorate(func, ray_opts)

    return CachedTask()
