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


def _cached_path_missing(cached: Any) -> Optional[Path]:
    """If `cached` represents a file path that no longer exists on disk,
    return the missing path; otherwise None.

    Handles bare Path / FitsFilePath / XisfFilePath / TiffFilePath
    (all subclass `Path`) plus `WithSidecar` wrappers around them, and
    any result type carrying a `.filename: Path` (e.g. ImageMetadata).
    Anything else (numpy arrays, dicts, plain strings, ContextRefs)
    is left alone — only file-path returns are validated.

    Lets the cache self-heal when an output file gets deleted out from
    under us — the next read on a stale entry is treated as a miss and
    the producer re-runs, rather than handing back a dangling path.
    """
    from file_io.types import WithSidecar

    if isinstance(cached, WithSidecar):
        cached = cached.path
    if not isinstance(cached, Path):
        # Result types like ImageMetadata wrap a path in `.filename`.
        # If that file is gone, the cached metadata is stale — fall
        # through to the missing-path branch.
        inner = getattr(cached, "filename", None)
        if isinstance(inner, Path):
            cached = inner
        else:
            return None
    return cached if not cached.exists() else None


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
        # Pull out the optional cache-key narrowing hook. Functions opt in
        # by passing `cache_key_narrow=...` to @node(); the hook receives
        # `(args, kwargs)` and returns `(args, kwargs)` to be hashed
        # *instead of* the actual call args. Function execution still
        # gets the full original args — only the cache key uses the
        # narrowed view. See docs in the per-function narrow modules.
        cache_key_narrow = ray_opts.pop("cache_key_narrow", None)
        func = _beartype(func)

        def _get_cache_key_and_result(bound):
            if cache_key_narrow is not None:
                hash_args, hash_kwargs = cache_key_narrow(bound.args, bound.kwargs)
            else:
                hash_args, hash_kwargs = bound.args, bound.kwargs
            call_hash = call_hasher(fn_hash, *hash_args, **hash_kwargs)
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

            missing = _cached_path_missing(cached)
            if missing is not None:
                debug_print(
                    f"cache hit on stale entry: {key} -> {missing} no longer "
                    f"on disk, treating as miss"
                )
                return key, call_hash, False, None

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
        ray_opts.setdefault("max_retries", 3)
        ray_opts.setdefault("retry_exceptions", True)

        ray_func = ray.remote(**ray_opts)(inner)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Implicit recording: inside a `@graph_builder` / `@pipeline`
            # trace, calling a @node op as plain Python emits a `Node`
            # handle into the active Graph instead of erroring. Outside
            # a trace, the original behaviour is preserved.
            from dependency_graph.tracing import _TRACE, record
            if _TRACE.get() is not None:
                # `_after` is a magic kwarg for ordering-only deps; pop
                # it before validating that user kwargs are absent.
                _after = kwargs.pop("_after", ())
                if kwargs:
                    raise NotImplementedError(
                        f"{func.__name__} called with kwargs inside a "
                        f"trace ({sorted(kwargs)}); only positional args "
                        "are supported in trace context."
                    )
                handle = record(wrapper, *args, _after=_after)
                # Generic post-record hooks: each `_trace_hook` is a
                # callable `Node -> Node` that records additional graph
                # nodes downstream of this call and returns the new
                # sink. Lets decorators like `@makes_image` chain on
                # their own canonical downstream ops (e.g.
                # `compute_image_metadata`) without this module knowing
                # what's being chained.
                for hook in getattr(wrapper, "_trace_hooks", ()):
                    handle = hook(handle)
                return handle
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
                return ray_func.options(memory=estimate).remote(*args, **kwargs)

            wrapper.remote = _auto_memory_remote
        else:
            wrapper.remote = ray_func.remote
        wrapper.options = ray_func.options
        wrapper._original = func
        wrapper._function = func
        wrapper._is_task = True

        # Forward all custom attributes from inner decorators (e.g.
        # filename_fn, filename_fns, _display, _trace_hooks) so they
        # survive the cached.remote wrapping.
        wrapper.__dict__.update(func.__dict__)

        # Re-register the @node-wrapped function under each identity
        # `func` registered for, so view-side enumeration via
        # producers_for(identity) gets the @node-wrapped callable
        # (with `.remote` and `_function`) instead of the inner
        # wrapper. Only @makes_image-decorated functions are in the
        # producer list — non-producer @node ops (e.g. graph builders
        # like calibrate) skip this step.
        if getattr(func, '_makes_image_wrapped', False):
            import products.registry as _registry_module
            for ops in _registry_module._producers.values():
                for inner in ops.values():
                    inner[:] = [wrapper if op is func else op for op in inner]

        def is_cached(
            *args, injected_context: Optional[dict] = None, **kwargs
        ) -> bool:
            bound = _bind_args(func, args, kwargs, injected_context)
            _, _, valid, _ = _get_cache_key_and_result(bound)
            return valid

        wrapper.is_cached = is_cached

        def invalidate(
            *args, injected_context: Optional[dict] = None, **kwargs
        ) -> bool:
            """Drop this call's cache entry. Returns True iff a key was deleted.

            Uses the same key-derivation as `is_cached` (including any
            cache_key_narrow hook), so callers can target the same entry
            without knowing the underlying hash.
            """
            bound = _bind_args(func, args, kwargs, injected_context)
            key, _, valid, _ = _get_cache_key_and_result(bound)
            if not valid:
                return False
            cache = _resolve_cache()
            try:
                del cache[key]
                return True
            except KeyError:
                return False

        wrapper.invalidate = invalidate

        def concerns(
            *args, injected_context: Optional[dict] = None, **kwargs
        ) -> list[dict]:
            """Concerns reported by the most recent run with these inputs.

            Looks up the per-object key written by `@capture_logs` when
            the wrapped function is a registered producer. Returns a
            (possibly empty) list of structlog event dicts; each carries
            `event` (title), `detail`, and `_concern` ("warning"/"error")
            plus any structured fields the caller attached. Returns []
            when the function isn't a registered producer.
            """
            from dependency_graph.logging import _concerns_keys_for
            bound = _bind_args(func, args, kwargs, injected_context)
            if cache_key_narrow is not None:
                hash_args, hash_kwargs = cache_key_narrow(bound.args, bound.kwargs)
            else:
                hash_args, hash_kwargs = bound.args, bound.kwargs
            keys = _concerns_keys_for(func, hash_args, hash_kwargs)
            cache = _resolve_cache()
            for key in keys:
                if key in cache:
                    return cache[key]
            return []

        wrapper.concerns = concerns

        return wrapper

    class CachedTask:
        def __call__(self, func: Callable) -> Callable:
            return decorate(func, ray_opts={})

        def remote(self, func: Callable = None, **ray_opts):
            if func is None:
                return lambda f: decorate(f, ray_opts)
            return decorate(func, ray_opts)

    return CachedTask()
