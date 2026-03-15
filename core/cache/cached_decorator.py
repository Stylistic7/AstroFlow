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
from hashing.types import LargeObject
from protocols import CacheProtocol, ContentHashProtocol
from cache.types import NeverRun


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
    content_hasher: ContentHashProtocol,
    call_hasher: Optional[Callable[[str, Any], str]],
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
            cached = _resolve_cache()[key]
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

            if isinstance(cached, LargeObject):
                assert cached.ref is None

            stored_hash = None
            try:
                stored_hash = content_hasher.get(cached)
            except KeyError:
                print(
                    f"### WARNING: Could not get content hash of object "
                    f"type {type(cached)}"
                )

            if not content_hasher.has(cached) or stored_hash == call_hash:
                debug_print(f"cache hit: {key}")
                return key, call_hash, True, cached
            else:
                debug_print(f"cache mismatch ({stored_hash}): {key}")
                return key, call_hash, False, None

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

            if valid and not isinstance(cached_result, Exception):
                if isinstance(cached_result, LargeObject):
                    assert cached_result.ref is None
                return cached_result

            if dry_run:
                return NeverRun

            try:
                start_time = time.time()
                result = func(*bound.args, **bound.kwargs)
                elapsed_time = time.time() - start_time
                if isinstance(result, ray.exceptions.RayTaskError):
                    print(
                        f"### fn {func.__name__} raised ray.exceptions.RayTaskError {result}"
                    )
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
            except Exception as e:
                result = e

            def _handle_object_wrappers(obj: Any) -> Any:
                assert not isinstance(obj, ray.ObjectRef)

                if isinstance(obj, LargeObject):
                    return obj.strip_ref()

                return obj

            result = content_hasher.set(result, call_hash)
            if isinstance(result, ray.exceptions.RayTaskError):
                print(
                    f"### fn {func.__name__} content_hasher raised ray.exceptions.RayTaskError {result}"
                )
            debug_print(
                f"storing: {key} as {type(result)} into cache {_resolve_cache().directory}"
            )
            _resolve_cache()[key] = _handle_object_wrappers(result)

            if isinstance(result, Exception):
                raise result
            else:
                return result

        inner.__name__ = func.__name__
        inner.__qualname__ = func.__qualname__
        inner.__module__ = func.__module__

        ray_func = (
            ray.remote(**ray_opts)(inner) if ray_opts else ray.remote(inner)
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            raise RuntimeError(
                f"{func.__name__} is a remote task. Use `.remote(...)`."
            )

        wrapper.remote = ray_func.remote
        wrapper.options = ray_func.options
        wrapper._original = func
        wrapper._function = func
        wrapper._is_task = True

        for attr in ("filename_fn", "thumbnail_fn"):
            if hasattr(func, attr):
                setattr(wrapper, attr, getattr(func, attr))

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
