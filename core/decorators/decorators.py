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
from profiling.measure_performance import make_measure_performance_decorator
from profiling.memory_estimate import estimate_memory, _FIXED_MEMORY_BY_OP
from dependency_graph.logging import make_capture_logs_decorator
from hashing import hash_call, hash_source_code
from context import context, get_cache


def _head_aware_cache():
    """`@node` cache provider: the head uses its local (canonical)
    diskcache directly — fast, and it IS the source of truth there; every
    other node routes through the cluster-wide `ContextActor` so no worker
    reads/writes a divergent local diskcache. Resolved once per process
    (`make_cached_decorator` memoizes the returned cache)."""
    from cache.cached_decorator import _is_head_node
    if _is_head_node():
        return get_cache()
    from cache.context_actor_cache import ContextActorCache
    return ContextActorCache()


cached = make_cached_decorator(
    cache_provider=_head_aware_cache,
    memory_estimator=estimate_memory,
    debug=False,
)

create_thumbnail = make_create_thumbnail_decorator(context)
create_tile_pyramid = make_create_tile_pyramid_decorator(context)

measure_performance = make_measure_performance_decorator(context)
capture_logs = make_capture_logs_decorator(context)


# File-typed return annotations that signal "this is an artifact
# producer." Listed locally rather than imported from products.registry
# so bare `@node` decoration paths don't have to load that module.
_FILE_RETURN_TYPE_NAMES = ('TiffFilePath', 'XisfFilePath', 'FitsFilePath')


def _looks_like_file_return(func) -> bool:
    """Return True iff `func`'s return annotation names a file-typed
    result directly (`-> XisfFilePath`) or as a generic arg
    (`-> WithSidecar[XisfFilePath]`)."""
    import inspect, typing
    raw = func.__wrapped__ if hasattr(func, '__wrapped__') else func
    try:
        sig = inspect.signature(raw)
    except (TypeError, ValueError):
        return False
    ret = sig.return_annotation
    if ret is inspect.Signature.empty:
        return False
    if getattr(ret, '__name__', None) in _FILE_RETURN_TYPE_NAMES:
        return True
    for a in typing.get_args(ret):
        if getattr(a, '__name__', None) in _FILE_RETURN_TYPE_NAMES:
            return True
    return False


def _metadata_wrap(func):
    """Run `func`, then synchronously extract image metadata from its
    return value and yield the resulting `ImageMetadata` to callers.

    Produced images are NEVER plate-solved here. Solving costs ~45 s on one
    core, and inside the producer's task it holds the producer's whole
    reservation (12 CPUs for a Siril stack, a GPU slot for BXT) while the
    rest sits idle. A pipeline that needs a solution says so in the graph:
    `metadata.solve_nodes.plate_solve_image` after an op that moves pixels,
    `inherit_plate_solution` after one that doesn't (starless / star layers).

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
        return compute_image_metadata._function(result, plate_solve=False)

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


def _assert_pure_of_context(func) -> None:
    """A @node may not touch `context` — that is hidden state.

    A node's cache key is computed from its ARGUMENTS. Anything the body
    reads that is not an argument is invisible to the key, so the cached
    result stops corresponding to the thing it is keyed on and no eviction
    can notice. Anything it WRITES is worse: on a cache hit the body never
    runs, so the side effect silently does not happen.

    Passing a `ContextRef` as a PARAMETER is the supported idiom and is
    pure — the ref carries its own content_hash, so the dependency is
    declared and enters the call hash. Builders that must touch context are
    `@ray.remote` (see `build_regroup_graph`), never `@node`.

    Raised at decoration (import) time, so a violation is a load failure
    rather than a wrong answer found hours into a run.
    """
    import ast
    import inspect
    import textwrap

    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    except (OSError, TypeError, SyntaxError, IndentationError):
        return  # dynamically built function: nothing to inspect
    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
         and n.name == func.__name__),
        None,
    )
    if fn is None:
        return
    params = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    if fn.args.vararg:
        params.add(fn.args.vararg.arg)
    if fn.args.kwarg:
        params.add(fn.args.kwarg.arg)
    for n in ast.walk(fn):
        if isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] == "context":
            offence = f"`from {n.module} import ...` on line {n.lineno}"
        elif isinstance(n, ast.Name) and n.id == "context" and "context" not in params:
            offence = f"a reference to `context` on line {n.lineno}"
        else:
            continue
        raise TypeError(
            f"@node on {func.__name__!r} touches context: {offence}. A node's "
            "cache key comes from its arguments, so context access is hidden "
            "state the key cannot see (and a WRITE is skipped entirely on a "
            "cache hit). Pass a ContextRef as a parameter instead — it carries "
            "a content_hash and enters the call hash — or make this an "
            "@ray.remote builder if it genuinely must touch context."
        )


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

        # Image producer: an op is an artifact producer iff its return
        # annotation resolves to a file-typed result (XISF / TIFF /
        # FITS). Routing on disk is
        # `<output_dir>/<op_name>/<call_hash>.<ext>`; the registry keys
        # by the type of the first parameter.
        @node(
            final=True,
            display="Master",
            cache_key_narrow=_narrow,
            resources={"PixInsight": 1},
            auto_memory=True,
        )
        def stack_files(
            stack: BaseStack, config, *cal_masters,
        ) -> WithSidecar[XisfFilePath]: ...
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
        # cached entry under the old wrapper is now orphaned and the
        # `@node`-decorated body never re-runs as-is. Warn loudly so the
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
        final = ray_opts.pop("final", False)
        display = ray_opts.pop("display", None)
        # Friendly graph label for the workflow UI ("Blend panels" instead of
        # `_blend_many_impl`). Pure presentation: the structural source hash
        # strips decorator lists, so declaring/editing this never re-keys.
        display_name = ray_opts.pop("display_name", None)
        # Declarative tap filter for deliverables that have NO `filter` signature
        # param and no per-filter input entity — i.e. colour composites, which
        # combine every channel into one image. Their tap would otherwise route
        # as `filter=None` and fall out of the filter-keyed dashboard/tap/tile
        # plumbing. A sentinel string (e.g. "RGB") lets them flow through it
        # exactly like a mono per-filter tap. `filter_of` reads this first.
        tap_filter = ray_opts.pop("filter", None)
        custom_filename_fn = ray_opts.pop("filename_fn", None)
        cache_key_narrow = ray_opts.pop("cache_key_narrow", None)
        # The kind of frame this op emits, stamped into the output's IMAGETYP
        # header — but only when the body didn't already write one (so the
        # shared stacker's calibration masters keep "Master Dark" etc.). The
        # default "light" makes `is_light_frame` uniformly true for produced
        # images.
        image_type = ray_opts.pop("image_type", "light")
        # This op's output is a WORKING VALUE, not a deliverable: reclaimed as
        # soon as the graph that consumes it completes. The author of the op is
        # the one who knows whether it is cheap to recompute, so the decision
        # lives HERE rather than in whichever pipeline happens to build a scope
        # around it (which is where `Graph(delete_intermediates=True)` used to
        # put it — a different file from the cost knowledge, and duplicated at
        # every scope site).
        #
        # Lifetime, precisely: when a graph completes, every ephemeral node in
        # it is reclaimed EXCEPT that graph's terminal, whose result escapes to
        # consumers outside. That is a DEFERRAL, not an exemption — one level
        # up the same node is interior, so the enclosing graph reclaims it.
        ephemeral = ray_opts.pop("ephemeral", False)
        # The op's result is an IDENTITY, not a derivation: its content hash
        # names the data it identifies (`run_from_frames`: the frames), so the
        # cache layer leaves it alone instead of stamping the call hash. Every
        # other entity-valued op is derived and hashes as its call.
        identity_root = ray_opts.pop("identity_root", False)
        # `declares_identity`: this op stamps its own CONTHASH, so its output's
        # identity is NOT its call hash (see `file_io.generic`). Both sides
        # that predict a consumer's key — `preflight._dep_identity` and the
        # trace-time handle stamp — read the declaration off disk for these
        # ops only, so the lookup stays a handful of files rather than one per
        # node. Opting in is what buys EARLY CUTOFF: rebuild under a new call
        # hash, write the same content, and nothing below re-keys.
        declares_identity = ray_opts.pop("declares_identity", False)

        # A `final=True` op is a deliverable, and a deliverable is always
        # user-facing (shown + openable). So `final` requires an explicit,
        # author-chosen `display="Name"`. Fail loud at decoration time.
        if final and display is None:
            raise ValueError(
                f"@node on {func.__name__!r}: final=True requires "
                'display="Name" (a final deliverable is always user-facing).'
            )
        # A deliverable that deletes itself is a contradiction, and the failure
        # would be silent and remote (a missing tap, hours later). Catch it at
        # decoration time instead.
        if final and ephemeral:
            raise ValueError(
                f"@node on {func.__name__!r}: final=True and ephemeral=True "
                "are contradictory — a deliverable is never reclaimed."
            )

        _assert_pure_of_context(func)

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

        # An op is an "artifact producer" iff it writes a file on disk:
        # either its return annotation resolves to a file-typed result
        # (TIFF, XISF, etc.), or it was given an explicit `filename_fn=`
        # naming where its file lands (selection-style ops that return a
        # `WithSidecar` pointing at a raw frame).
        #
        # Checked inline (rather than calling into `products.registry`)
        # so a bare @node decoration on a non-artifact op never has
        # to import the registry — avoids a circular-import chain
        # during module load of `clustering/` / similar bare-@node
        # modules.
        is_artifact_producer = (
            _looks_like_file_return(func) or custom_filename_fn is not None
        )

        filename_fn = None
        if is_artifact_producer:
            from products.registry import build_filename_fn, fully_resolve
            if custom_filename_fn is not None:
                # Caller passed an explicit `filename_fn=` (rare, used
                # by selection-style ops that return absolute paths to
                # raw frames). Wrap with `fully_resolve` so the routing
                # prefix (`<output_dir>/<op_name>/`) still gets prepended
                # for relative paths.
                filename_fn = fully_resolve(custom_filename_fn, func.__name__)
            else:
                filename_fn = build_filename_fn(func, canonical_call_hash)

        # ---- 3. Compose layers ----------------------------------------
        # Each layer is single-concern. None of them queries another's
        # state via attribute lookup — coordination flows through the
        # explicit arguments below.
        wrapped = func

        if is_artifact_producer:
            from products.registry import makes_image_wrap
            wrapped = makes_image_wrap(
                wrapped, canonical_call_hash=canonical_call_hash,
                # A tile-pyramid TIFF (the web-openable co-representation) is
                # the human-facing artifact — built only for `display=` ops.
                # Non-display intermediates write just their primary. (An
                # ephemeral `display=` op still builds its TIFF; the ephemeral
                # cleanup deletes the whole atom — primary + TIFF — together.)
                makes_tiff=display is not None,
                image_type=image_type,
            )
            wrapped = _metadata_wrap(wrapped)

        wrapped = capture_logs(wrapped, cache_key_narrow=cache_key_narrow)
        wrapped = measure_performance(wrapped, ray_opts=ray_opts)

        # Isolate PROFILED auto_memory ops in a fresh worker so the absolute
        # peak-RSS that measure_performance records (see measure_performance.py)
        # reflects THIS op only. Ray reuses CPU task workers across ops, so
        # without this a small op inherits a big op's resident footprint and its
        # memory= reservation balloons — over-reserving and serializing the pool.
        # max_calls=1 makes the worker EXIT after one task -> a fresh worker per
        # invocation. Excluded: ops with a FIXED reservation (_FIXED_MEMORY_BY_OP)
        # never consult the profile and don't need measurement — and
        # compute_image_metadata in particular is dispatched ~30k times, so it
        # must keep worker reuse (a ~2-4s respawn each would dominate). max_calls
        # is task-only and cannot be set via .options(), so it's baked here at
        # ray.remote decoration time (Ray preserves it through .options(memory=)).
        if ray_opts.get("auto_memory") and func.__name__ not in _FIXED_MEMORY_BY_OP:
            ray_opts["max_calls"] = 1

        wrapper = cached.remote(
            wrapped,
            canonical_call_hash=canonical_call_hash,
            identity_root=identity_root,
            declares_identity=declares_identity,
            **ray_opts,
        )

        # ---- 4. Attach public-API attributes on the final wrapper -----
        wrapper._fn_hash = fn_hash
        wrapper._cache_key_narrow = cache_key_narrow
        # Read by the runtime's reclaim pass at every graph completion. Set
        # unconditionally (not only for artifact producers) so the flag reads
        # the same on every op; a node that produces no file simply has nothing
        # to reclaim.
        wrapper._ephemeral = ephemeral
        if display_name is not None:
            wrapper._display_name = display_name

        if is_artifact_producer:
            wrapper.filename_fn = filename_fn
            wrapper._final = final
            if display is not None:
                wrapper._display = display
            if tap_filter is not None:
                wrapper._tap_filter = tap_filter
            from products.registry import register_producer
            register_producer(wrapper, func)

        return wrapper


def _normalize_identities(identity):
    """Accept None, a single identity class, or a list — return a list."""
    if identity is None:
        return []
    if isinstance(identity, (list, tuple)):
        return list(identity)
    return [identity]


node = _Node()
