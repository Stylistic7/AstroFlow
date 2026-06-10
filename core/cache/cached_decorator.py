"""Single-concern: cache a function's results.

`cached.remote(func, *, canonical_call_hash, **ray_opts)` builds a Ray-
remote wrapper that:

1. Computes a cache key from args via `canonical_call_hash(*args, **kwargs)`.
2. Returns the cached result if present.
3. Otherwise runs `func`, stores the result under that key, and returns it.

That's the entire concern. There is no filename construction, no content-
hash stamping, no identity routing, no narrowing logic — all of those are
image-lifecycle concerns owned elsewhere (`makes_image_wrap`) and
orchestrated by `@node`, which passes the resulting `canonical_call_hash`
closure in.

This file used to interleave caching with the call-hash construction and
the on-disk filename_fn build. Those were folded out so that this module
can't silently disagree with the filename about which call_hash it's
using — the only call_hash that exists is the one its caller provides.
"""
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

from cache.context_ref import ContextRef, strip_refs
from cache.types import NeverRun
from protocols import CacheProtocol


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

    Lets the cache self-heal when an output file gets deleted out from
    under us — the next read on a stale entry is treated as a miss.
    """
    from file_io.types import WithSidecar
    if isinstance(cached, WithSidecar):
        cached = cached.path
    if not isinstance(cached, Path):
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


def make_cached_decorator(
    cache_provider: Callable[[], CacheProtocol],
    memory_estimator: Optional[Callable[[Callable, tuple, dict], int]] = None,
    debug: bool = False,
):
    """Build a `cached` object. Use `cached.remote(func, canonical_call_hash=...)`
    to decorate a function. The caller (`@node`) computes
    `canonical_call_hash` and passes it in; this module never builds one
    of its own, eliminating the possibility of two competing call_hash
    definitions for the same op."""
    cache: CacheProtocol | None = None

    def _resolve_cache():
        nonlocal cache
        if cache is None:
            cache = cache_provider()
        return cache

    def debug_print(*args, **kwargs):
        if debug:
            print(*args, **kwargs)

    def decorate(
        func: Callable,
        *,
        canonical_call_hash: Callable,
        ray_opts: dict,
        needs_metadata_upgrade: bool = False,
    ) -> Callable:
        """Wrap `func` with caching keyed by `canonical_call_hash(args)`.

        `needs_metadata_upgrade=True` opts in to the on-read
        WithSidecar→ImageMetadata cache shim — see the TODO(cache-wipe)
        block in _key_and_lookup."""
        func = _beartype(func)

        def _key_and_lookup(bound):
            call_hash = canonical_call_hash(*bound.args, **bound.kwargs)
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
            # TODO(cache-wipe): remove this shim — and the
            # `needs_metadata_upgrade` decorate-kwarg + its plumbing —
            # the next time the diskcache is wiped (see
            # `~/.claude/projects/.../memory/project_cache_wipe_metadata_migration.md`).
            # Pre-2026-06-09 cache entries for `@node(identity=...)` ops
            # were `WithSidecar[Path]`; post-refactor they're
            # `ImageMetadata` (the metadata extraction was folded into
            # the producer's Ray task). Upgrade old-shape entries on
            # first read so consumers always see the new shape.
            if needs_metadata_upgrade:
                from file_io.types import WithSidecar
                if isinstance(cached, WithSidecar):
                    # File-existence check BEFORE the upgrade.
                    # `compute_image_metadata` opens the file to derive
                    # content_hash; if the file was reclaimed by an
                    # earlier ephemeral cleanup, the upgrade would
                    # raise FileNotFoundError and kill the task.
                    # Treat as a stale miss instead, same as the
                    # post-upgrade check below — the producer will be
                    # re-run on the miss path and write the new shape
                    # natively.
                    pre_missing = _cached_path_missing(cached)
                    if pre_missing is not None:
                        debug_print(
                            f"cache hit on stale pre-upgrade entry: {key} -> "
                            f"{pre_missing} no longer on disk, treating as miss"
                        )
                        return key, call_hash, False, None
                    from metadata.image_metadata import compute_image_metadata
                    cached = compute_image_metadata._function(cached)
                    _resolve_cache()[key] = strip_refs(cached)
            size_mb = objsize.get_deep_size(cached) / 1024.0 / 1024.0
            if size_mb > 5:
                debug_print(
                    f"### cache key {key} contained large object, {size_mb:0.3f} MB"
                )
            missing = _cached_path_missing(cached)
            if missing is not None:
                debug_print(
                    f"cache hit on stale entry: {key} -> {missing} no "
                    f"longer on disk, treating as miss"
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
            key, call_hash, valid, cached_result = _key_and_lookup(bound)
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
                f"storing: {key} as {type(result)} into cache "
                f"{_resolve_cache().directory}"
            )
            _resolve_cache()[key] = strip_refs(result)
            return result

        inner.__name__ = func.__name__
        inner.__qualname__ = func.__qualname__
        inner.__module__ = func.__module__

        auto_memory = ray_opts.pop("auto_memory", False)
        ray_opts.setdefault("scheduling_strategy", "SPREAD")
        ray_opts.setdefault("max_retries", 3)
        ray_opts.setdefault("retry_exceptions", True)
        ray_func = ray.remote(**ray_opts)(inner)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Trace-mode recording: inside a `@graph_builder` / `@pipeline`
            # trace, calling a @node op as plain Python emits a `Node`
            # handle into the active Graph instead of erroring.
            from dependency_graph.tracing import _TRACE, record
            from cache.ephemeral_scope import _EPHEMERAL_SCOPE
            if _TRACE.get() is not None:
                _after = kwargs.pop("_after", ())
                ephemeral = bool(kwargs.pop("_ephemeral", _EPHEMERAL_SCOPE.get()))
                if kwargs:
                    raise NotImplementedError(
                        f"{func.__name__} called with kwargs inside a "
                        f"trace ({sorted(kwargs)}); only positional args "
                        "are supported in trace context."
                    )
                handle = record(wrapper, *args, _after=_after)
                if ephemeral:
                    if getattr(wrapper, "_final", False):
                        # A `final=True` op (a `store_*` terminal) is
                        # always a deliverable. Wrapping one in an
                        # `ephemeral_intermediates()` block is always a
                        # user bug — the terminal-predecessor topology
                        # guard in the old `_maybe_clean_predecessors`
                        # protected this implicitly; with the global
                        # registry an unconsumed terminal would just
                        # never get registered (no consumers) so it'd
                        # never clean anyway, but a downstream consumer
                        # appearing later would silently unlink the
                        # final output. Refuse explicitly at trace time.
                        raise RuntimeError(
                            f"{func.__name__} is `final=True` (a deliverable "
                            "terminal) but was called inside "
                            "`ephemeral_intermediates()`. Final outputs "
                            "must never be marked ephemeral — either drop "
                            "the `final=True` or move the call out of the "
                            "ephemeral scope."
                        )
                    handle._ephemeral = True
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

        def is_cached(*args, injected_context=None, **kwargs) -> bool:
            bound = _bind_args(func, args, kwargs, injected_context)
            _, _, valid, _ = _key_and_lookup(bound)
            return valid
        wrapper.is_cached = is_cached

        def invalidate(*args, injected_context=None, **kwargs) -> bool:
            """Drop this call's cache entry. Returns True iff a key was deleted.

            Bypasses `_key_and_lookup` because that function's stale-miss
            check (`_cached_path_missing`) returns valid=False when the
            on-disk file is already gone. Ephemeral cleanup calls
            `invalidate` AFTER unlinking the file, so the lookup would
            always report "nothing to invalidate" and the cache entry
            would leak. Just compute the key and unconditionally `del`.
            """
            bound = _bind_args(func, args, kwargs, injected_context)
            call_hash = canonical_call_hash(*bound.args, **bound.kwargs)
            key = (func.__name__, call_hash)
            cache = _resolve_cache()
            try:
                del cache[key]
                return True
            except KeyError:
                return False
        wrapper.invalidate = invalidate

        return wrapper

    class CachedTask:
        """Use `cached.remote(func, canonical_call_hash=..., **ray_opts)`
        to wrap a function. The caller (`@node`) is responsible for
        constructing `canonical_call_hash`.

        For test code and other direct callers, `canonical_call_hash` may
        be omitted (or `None`); a default that hashes
        `(hash_source_code(func), *args, **kwargs)` with no narrowing is
        built automatically. Production code always goes through `@node`,
        which supplies an explicit `canonical_call_hash` that includes
        the op's narrow hook.
        """

        def remote(self, func=None, *, canonical_call_hash=None, cache_key_narrow=None,
                   needs_metadata_upgrade=False, **ray_opts):
            def go(f):
                cch = canonical_call_hash or _default_canonical_call_hash(f, cache_key_narrow)
                return decorate(f, canonical_call_hash=cch, ray_opts=ray_opts,
                                needs_metadata_upgrade=needs_metadata_upgrade)
            if func is None:
                return go
            return go(func)

    return CachedTask()


def _default_canonical_call_hash(func, cache_key_narrow):
    """Build a default `canonical_call_hash` for direct `cached.remote`
    callers that didn't supply one. Production code (`@node`) builds its
    own; this is a convenience for tests and stand-alone usage."""
    from hashing import hash_call, hash_source_code
    fn_hash = hash_source_code(func)
    def canonical_call_hash(*args, **kwargs):
        if cache_key_narrow is not None:
            args, kwargs = cache_key_narrow(args, kwargs)
        return hash_call(fn_hash, *args, **kwargs)
    return canonical_call_hash
