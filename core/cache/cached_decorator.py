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

from cache.content_hashed import ContentHashed
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


_MISSING = object()  # sentinel: a cache miss, distinct from a stored None


def _is_head_node() -> bool:
    """True iff this process runs on the node carrying the canonical
    diskcache. Detected via the head-only `diskcache` custom resource —
    the same invariant the orchestrator actors pin on. (The prior
    `node:__internal_head__` marker isn't reliably published in this Ray
    version; see ray_init.py.) Returns False if the node can't be located,
    which safely routes cache access to the canonical ContextActor."""
    node_id = get_runtime_context().get_node_id()
    for n in ray.nodes():
        if n["NodeID"] == node_id:
            return "diskcache" in n.get("Resources", {})
    return False


# Whether THIS worker process runs on the head node that owns the canonical
# diskcache. Constant for a process's lifetime, so compute it once (the check
# does a `ray.nodes()` RPC) and reuse.
_PROCESS_IS_HEAD: Optional[bool] = None


def _log_node_exec(
    op: str,
    call_hash: str,
    outcome: str,
    elapsed_s: Optional[float] = None,
    result: Any = None,
    size_mb: Optional[float] = None,
) -> None:
    """Emit one structured `### node_exec` record per real (non-dry-run) node
    call. Grepping/parsing `### node_exec` then surfaces a broad class of
    execution problems, not just today's re-run hunt:

        field      what it catches
        op         which operation ran.
        call_hash  the cache key. Same (op, call_hash) with outcome=miss more
                   than once => a cache-miss RE-RUN (memo not hit). Many
                   *distinct* call_hashes for one op => upstream content-hash
                   INSTABILITY (a determinism leak feeding the trace).
        outcome    hit | miss — the cache hit rate; a low rate is wasted compute.
        node,pid   which machine / worker ran it. Two misses of one call_hash on
                   different `node`s => per-machine diskcache divergence
                   (HEAD-PINNING); `pid` isolates a single bad worker.
        head       ran on the head (canonical-cache) node or not — off-head
                   misses that then re-run elsewhere are the head-pinning smell.
        elapsed_s  run time — outliers flag slow ops, hangs, perf regressions.
        result_mb  result size — outliers flag memory bloat / oversized cache.
        rtype,out  result type and output path — a miss whose `out` file is then
                   absent flags a producer that didn't write, or reclaim churn.

    Never raises — telemetry must never break the pipeline."""
    global _PROCESS_IS_HEAD
    try:
        node = str(get_runtime_context().get_node_id())[:12]
    except Exception:
        node = "?"
    if _PROCESS_IS_HEAD is None:
        try:
            _PROCESS_IS_HEAD = _is_head_node()
        except Exception:
            _PROCESS_IS_HEAD = False
    out = "-"
    try:
        r = result
        from file_io.types import WithSidecar
        if isinstance(r, WithSidecar):
            r = r.path
        if isinstance(r, Path):
            out = str(r)
        elif getattr(r, "filename", None) is not None:
            out = str(r.filename)
    except Exception:
        pass
    print(
        f"### node_exec op={op} call_hash={call_hash} outcome={outcome} "
        f"node={node} pid={os.getpid()} head={_PROCESS_IS_HEAD} "
        f"elapsed_s={f'{elapsed_s:.3f}' if elapsed_s is not None else '-'} "
        f"result_mb={f'{size_mb:.2f}' if size_mb is not None else '-'} "
        f"rtype={type(result).__name__ if result is not None else '-'} out={out}"
    )


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
    ) -> Callable:
        """Wrap `func` with caching keyed by `canonical_call_hash(args)`."""
        func = _beartype(func)

        def _key_and_lookup(bound):
            call_hash = canonical_call_hash(*bound.args, **bound.kwargs)
            key = (func.__name__, call_hash)
            start_time = time.time()
            # One cache op (one RPC when the cache is the ContextActor
            # proxy off-head): `.get` with a process-local sentinel, so a
            # miss is distinguished without a separate `__contains__` hit.
            cached_raw = _resolve_cache().get(key, _MISSING)
            elapsed_time = time.time() - start_time
            if cached_raw is _MISSING:
                debug_print(f"cache miss: {key}")
                return key, call_hash, False, None
            cached = strip_refs(cached_raw)
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
                if not dry_run:  # dry-run tracing isn't real work — don't log it
                    _log_node_exec(func.__name__, call_hash, "hit",
                                   result=cached_result)
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
            size_mb = objsize.get_deep_size(result) / 1024.0 / 1024.0
            _log_node_exec(func.__name__, call_hash, "miss",
                           elapsed_s=elapsed_time, result=result, size_mb=size_mb)
            if TRACE_TASKS:
                print(f"<<< {func.__name__} ({elapsed_time:.3f}s)")
            if elapsed_time > 0.5:
                debug_print(
                    f"### fn {func.__name__} was slow to run, took {elapsed_time=} s"
                )
            if size_mb > 5:
                debug_print(
                    f"### fn {func.__name__} returned large object, {size_mb:0.3f} MB"
                )
            if isinstance(result, Path) and not result.exists():
                raise FileNotFoundError(result)
            # Pure-Merkle stamp for PLAIN-OBJECT results. The trace side already
            # stamps every handle with `content_hash = canonical_call_hash(...)`
            # and documents that as "the content_hash its runtime result will
            # carry" — but only image producers actually carried it (via the
            # CONTHASH header that `compute_image_metadata` reads back). A
            # ContentHashed value returned by a plain @node came back with
            # content_hash="", so a trace-time key built from the HANDLE and a
            # runtime key built from the RESULT disagreed — exactly the
            # mismatch that would break `process_in_panels`-style identity
            # prediction over a Viewbox handle. Results that set their own
            # content_hash (Stack, Target, Mosaic from the identify ops) are
            # left alone.
            if isinstance(result, ContentHashed) and not result.content_hash:
                object.__setattr__(result, "content_hash", call_hash)
            from dependency_graph.classes import Graph
            if isinstance(result, Graph):
                # A Graph is a PLAN, not a result: a description of work still
                # to dispatch, valid only inside the runtime pass that receives
                # it. Caching one persists dead handles across processes and
                # class revisions — a `__slots__` change on Graph made every
                # such pickle unreadable (2026-08-19, process_stacks_for_ui),
                # and even a readable one would replay a stale plan instead of
                # recomputing it against current state. Delta-only builders
                # (return dict on all-hits, Graph on misses) still get their
                # fast path: the all-hits DICT is cached; a miss pass re-runs
                # the cheap probe next startup, by which time its members hit.
                return result
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
        # ONE retry, exceptions included: enough to survive a transient
        # (worker death, wedged-PI kill, rc-astro flake) without tripling
        # the cost of a deterministic failure — a failing stack_files
        # re-paid its full WBPP on every Ray retry (vdB15, 2026-08-21).
        # Across drains a failed node re-dispatches anyway, so persistent
        # failures still surface every run; they just stop multiplying
        # within one.
        ray_opts.setdefault("max_retries", 1)
        ray_opts.setdefault("retry_exceptions", True)
        ray_func = ray.remote(**ray_opts)(inner)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Trace-mode recording: inside a `@graph_builder` / `@pipeline`
            # trace, calling a @node op as plain Python emits a `Node`
            # handle into the active Graph instead of erroring.
            from dependency_graph.tracing import _TRACE, record
            if _TRACE.get() is not None:
                _after = kwargs.pop("_after", ())
                if kwargs:
                    raise NotImplementedError(
                        f"{func.__name__} called with kwargs inside a "
                        f"trace ({sorted(kwargs)}); only positional args "
                        "are supported in trace context."
                    )
                handle = record(wrapper, *args, _after=_after)
                # Stamp the handle with its call hash — the pure-Merkle
                # content_hash its runtime result will carry (the CONTHASH
                # stamp IS the producer's call_hash). Computable inductively:
                # child handles in `args` already carry theirs, and
                # `_maybe_use_content_hash` resolves them. This makes a
                # handle hashable ANYWHERE — including direct filename_fn
                # calls at trace time (process_in_panels derives its panel
                # manifest that way) — instead of leaking an address-based
                # Node repr into the hash, which made panel manifests
                # per-process nondeterministic and silently broke the
                # eager-trace/lazy-trace bit-identity claim.
                try:
                    handle.content_hash = canonical_call_hash(*args)
                except Exception as e:
                    # LOUD, deliberately: an unstamped handle silently
                    # degrades every downstream path prediction (the
                    # structural resolver and filename_fn arithmetic both
                    # read this stamp), and the divergence LOOKS
                    # deterministic while disagreeing with the keys the run
                    # lands. The un-hashable arg itself still surfaces at
                    # dispatch.
                    print(f"### handle-stamp failed for {func.__name__}: "
                          f"{type(e).__name__}: {e}")
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
            # Exposed so the graph executor can predict this node's peak-RSS
            # estimate BEFORE submitting — the memory-aware admission gate uses
            # it to hold a heavy task until the machine actually has room (see
            # dependency_graph._estimate_node_memory). Signature mirrors the
            # call above: estimator(func, args, kwargs) -> bytes.
            wrapper._auto_memory_estimator = memory_estimator
        else:
            wrapper.remote = ray_func.remote
            wrapper._auto_memory_estimator = None

        wrapper.options = ray_func.options
        wrapper._original = func
        wrapper._function = func
        wrapper._is_task = True
        # The op's declared *custom* resources (e.g. {"PixInsight": 1}), exposed
        # so the graph executor can do resource-aware admission control without
        # knowing any resource name. `@node` is built on top of `cached.remote`,
        # so this single assignment covers both `@node` and direct
        # `@cached.remote` ops.
        wrapper._ray_resources = dict(ray_opts.get("resources") or {})

        def is_cached(*args, injected_context=None, **kwargs) -> bool:
            bound = _bind_args(func, args, kwargs, injected_context)
            _, _, valid, _ = _key_and_lookup(bound)
            return valid
        wrapper.is_cached = is_cached

        def cached_value(*args, injected_context=None, **kwargs):
            """(hit, value): a read-only probe of this call's cache entry —
            never executes the body. `hit` follows the same validity rules as
            a real lookup (including the stale-on-disk check), so a builder
            can collect hit values and emit graph nodes ONLY for misses —
            the pattern that keeps unchanged work out of the graph entirely."""
            bound = _bind_args(func, args, kwargs, injected_context)
            _, _, valid, cached = _key_and_lookup(bound)
            return (valid, cached if valid else None)
        wrapper.cached_value = cached_value

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
                   **ray_opts):
            def go(f):
                cch = canonical_call_hash or _default_canonical_call_hash(f, cache_key_narrow)
                return decorate(f, canonical_call_hash=cch, ray_opts=ray_opts)
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
