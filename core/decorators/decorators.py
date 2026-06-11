"""The single `@node` decorator.

`@node` is the only public decorator for declaring a cached, Ray-remote
op. It accepts every knob — Ray options, narrowing, image-producer
declarations — as keyword arguments and orchestrates the inner layers
by computing every shared artifact ONCE and passing them in as explicit
arguments. The inner layers don't read each other's attributes — every
coordination point is a function parameter.

Concerns split:

  * `cached.remote` (cache/cached_decorator.py): caching only. Receives
    `canonical_call_hash` and uses it for the cache key. Doesn't know
    about filenames, content_hashes, identities.

  * `makes_image_wrap` (products/registry.py): image lifecycle only.
    Wraps the function so that after the body returns, the file's
    `content_hash` header is stamped and presentations are generated.
    Doesn't touch the cache.

  * `@node` (this file): the sole orchestrator. Parses kwargs once,
    computes shared artifacts (`fn_hash`, `canonical_call_hash`,
    `filename_fns`), composes the layers passing artifacts as args,
    attaches public-API attributes (`filename_fn`, `_identities`, etc.)
    on the final wrapper, registers in `_producers`.

The only logic that spans multiple decorators lives here.
"""
from beartype import beartype as _beartype

from cache.cached_decorator import make_cached_decorator
from previews.create_thumbnail import make_create_thumbnail_decorator
from previews.create_tile_pyramid import make_create_tile_pyramid_decorator
from previews.create_video import make_create_video_decorator
from profiling.measure_performance import make_measure_performance_decorator
from profiling.memory_estimate import estimate_memory
from dependency_graph.logging import make_capture_logs_decorator
from hashing import hash_call, hash_source_code
from context import context, get_cache

cached = make_cached_decorator(
    cache_provider=get_cache,
    memory_estimator=estimate_memory,
    debug=False,
)

create_thumbnail = make_create_thumbnail_decorator(context)
create_tile_pyramid = make_create_tile_pyramid_decorator(context)
create_video = make_create_video_decorator(context)

measure_performance = make_measure_performance_decorator(context)
capture_logs = make_capture_logs_decorator(context)


def _metadata_wrap(func):
    """Run `func`, then synchronously extract image metadata from its
    return value and yield the resulting `ImageMetadata` to callers.

    Folds the `compute_image_metadata` step into every
    `@node(identity=...)` producer in place of the trace-time chain-
    hook Node that used to live between the producer and its
    consumers. The chain-hook approach raced with the runtime's
    `_maybe_clean_predecessors` (the chain-hook node was the
    producer's immediate dependent and finished in milliseconds,
    firing ephemeral cleanup before the real consumer ran);
    composing the wrap into the producer's own Ray task eliminates
    the indirection — the producer's only graph-level dependents are
    the actual file consumers.
    """
    import functools
    from metadata.classes import ImageMetadata

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        from metadata.image_metadata import compute_image_metadata
        return compute_image_metadata._function(result)

    # `functools.wraps` copies the wrapped function's `__annotations__`,
    # including its `WithSidecar[...]` return annotation. After this
    # wrap the actual return type is `ImageMetadata`, so override the
    # `return` annotation; otherwise the outer `_beartype` (in
    # `cached.remote`) rejects the call with a return-type violation.
    wrapper.__annotations__ = {
        **wrapper.__annotations__,
        "return": ImageMetadata,
    }
    return wrapper


class _Node:
    """Single decorator for cached Ray-remote ops.

    Examples:

        # Pure cached compute (no on-disk file).
        @node
        def helper(stack): ...

        @node(num_cpus=0.25)
        def small_compute(x): ...

        @node(resources={"PixInsight": 1}, scheduling_strategy="SPREAD")
        def pi_op(...): ...

        # Image producer (replaces the old `@makes_image` decorator).
        @node(
            identity=StackId,
            final=True,
            display="Master",
            cache_key_narrow=_narrow,
            resources={"PixInsight": 1},
            auto_memory=True,
        )
        def stack_files(stack, config, *cal_masters): ...

        # Polymorphic identity (one op produces files routed under two
        # ObjectType directories — e.g. stack_files for Stack and
        # CalibrationStack).
        @node(
            identity=[StackId, CalibrationStackId],
            cache_key_narrow=_narrow,
            final=True,
        )
        def stack_files(...): ...
    """

    def __call__(self, func=None, **kwargs):
        if func is not None:
            # Bare `@node` — `func` is the decorated callable.
            return self._wrap(func, {})
        # `@node(**kwargs)` — return a decorator.
        return lambda f: self._wrap(f, kwargs)

    def _wrap(self, func, all_kwargs):
        # ---- 0. Double-wrap guard -------------------------------------
        # `@node` wrapping an already-`@node`-wrapped function (e.g.
        # `my_op = node(...)(some_existing_op)`) builds a NEW wrapper
        # with a fresh fn_hash → fresh call_hash → fresh filename. Every
        # cached entry under the old wrapper is now orphaned, the
        # `@node`-decorated body never re-runs as-is, and `_record_pipeline_taps`
        # sees a new producer with no display. Warn loudly so the
        # pipeline author can either rename or drop the redundant wrap.
        if hasattr(func, "_fn_hash"):
            import warnings
            inner_name = getattr(
                getattr(func, "_function", func), "__name__", repr(func),
            )
            warnings.warn(
                f"@node is wrapping an already-@node-wrapped op "
                f"({inner_name}). The new wrapper will have its own "
                "fn_hash, breaking cache dedup with the original. Use "
                "the original directly, or rename and accept the extra "
                "wrap-layer's cost.",
                stacklevel=3,
            )

        # ---- 1. Parse kwargs ------------------------------------------
        ray_opts = dict(all_kwargs)
        identity = ray_opts.pop("identity", None)
        final = ray_opts.pop("final", False)
        display = ray_opts.pop("display", None)
        media_type = ray_opts.pop("media_type", None)
        custom_filename_fn = ray_opts.pop("filename_fn", None)
        cache_key_narrow = ray_opts.pop("cache_key_narrow", None)
        identities = _normalize_identities(identity)
        # Default media_type = IMAGE; only resolve when there are
        # identities to register (avoids pulling the products.registry
        # module — and its mosaics/targets/stacks transitive imports —
        # into the import path of every bare `@node` decoration).
        if identities and media_type is None:
            from products.registry import MediaType
            media_type = MediaType.IMAGE

        # ---- 2. Compute shared artifacts ------------------------------
        fn_hash = hash_source_code(func)

        def canonical_call_hash(*args, **kwargs):
            """The op's call_hash. Used by both the cache layer (cache
            key + content_hash stamp) and the on-disk filename_fn. They
            share THIS closure — same `cache_key_narrow`, same `fn_hash`
            — by construction, so drift between cache key and filename
            is structurally impossible."""
            if cache_key_narrow is not None:
                args, kwargs = cache_key_narrow(args, kwargs)
            return hash_call(fn_hash, *args, **kwargs)

        filename_fn = None
        filename_fns = None
        if identities:
            from products.registry import (
                validate_identities, build_filename_fns, fully_resolve,
            )
            validate_identities(func, identities)
            if custom_filename_fn is not None:
                # Caller passed an explicit `filename_fn=` (rare, used by
                # selection-style ops that return absolute paths to raw
                # frames). Apply it per identity; skip the call_hash
                # convention entirely.
                filename_fns = {
                    identity: fully_resolve(
                        identity, custom_filename_fn, func.__name__,
                    )
                    for identity in identities
                }
                from products.registry import _make_dispatcher
                filename_fn = _make_dispatcher(filename_fns)
            else:
                filename_fns, filename_fn = build_filename_fns(
                    func, identities, canonical_call_hash,
                )

        # ---- 3. Compose layers ----------------------------------------
        # Each layer is single-concern. None of them queries another's
        # state via attribute lookup — coordination flows through the
        # explicit arguments below.
        wrapped = func

        if identities:
            from products.registry import makes_image_wrap
            wrapped = makes_image_wrap(
                wrapped,
                canonical_call_hash=canonical_call_hash,
                identities=identities,
            )
            wrapped = _metadata_wrap(wrapped)

        wrapped = capture_logs(wrapped, cache_key_narrow=cache_key_narrow)
        wrapped = measure_performance(wrapped, ray_opts=ray_opts)
        wrapper = cached.remote(
            wrapped,
            canonical_call_hash=canonical_call_hash,
            **ray_opts,
        )

        # ---- 4. Attach public-API attributes on the final wrapper -----
        # These are the ATTRIBUTES users access on the wrapper. The
        # inner layers don't read them — they had everything they needed
        # passed in as args. These exist purely so external code (user
        # pipelines, the trace, the runtime, the view-side enumeration)
        # can introspect the op.
        wrapper._fn_hash = fn_hash
        wrapper._cache_key_narrow = cache_key_narrow

        if identities:
            wrapper.filename_fn = filename_fn
            wrapper.filename_fns = filename_fns
            wrapper._identities = identities
            wrapper._final = final
            if display is not None:
                wrapper._display = display
            from products.registry import register_producer
            register_producer(wrapper, identities, media_type=media_type)

        return wrapper


def _normalize_identities(identity):
    """Accept None, a single identity class, or a list — return a list."""
    if identity is None:
        return []
    if isinstance(identity, (list, tuple)):
        return list(identity)
    return [identity]


node = _Node()
