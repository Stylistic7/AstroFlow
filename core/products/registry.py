import dataclasses
import inspect
import os
import tempfile
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Any, Type, Optional, Union

import numpy as np

from file_io.types import ContentHashable
from previews.types import PIXINSIGHT, ALL_QUALITY_LEVELS, ALL_IMPLEMENTATIONS, DOWNSCALED_MASTER,MASTER
from products.identity import ObjectId, PresentationType
from context import context


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


type FilenameFn = Union[str, Callable[..., Path]]


# LEGACY producer registry — kept as a backwards-compat layer for the
# small set of view-side endpoints that haven't yet migrated to the
# pipeline+tap model. New code SHOULD NOT call register_producer /
# get_producer / get_filename_fn:
#
# * Internal op-to-op lookups: import the producing op directly, use
#   `op.filename_fn(...)` (auto-dispatched across registered identities)
#   or `op.filename_fns[(identity, level)](...)` (explicit per-mode).
# * View-side lookups for "where did pipeline X publish?": go through
#   the PipelineOutputIndex (`pipelines/output_index.py`), populated
#   at pipeline-trace time by walking @makes_image(display=...) Nodes.
#
# The dicts below are still written by `@makes_image`'s decoration
# loop; deletion is gated on retiring the remaining (impl, level)-
# keyed view-side endpoints (/api/object_versions etc.).
_registry: dict[
    MediaType, dict[tuple[Type, str, str], Callable[..., Path]]
] = {
    MediaType.IMAGE: {},
    MediaType.VIDEO: {},
}

_filename_fns: dict[
    MediaType, dict[tuple[Type, str, str], Callable[..., Path]]
] = {
    MediaType.IMAGE: {},
    MediaType.VIDEO: {},
}


def register_producer(
    media_type: MediaType,
    level: str,
    func: Callable[..., Path],
    filename_fn: "FilenameFn | None" = None,
    implementation: str = PIXINSIGHT,
    identity: "type[ObjectId] | None" = None,
) -> None:
    """Legacy: register a producer under (identity, level, implementation).
    Internal-only — only `@makes_image` decoration writes to the registry."""
    key = (identity, level, implementation)
    _registry[media_type][key] = func
    if filename_fn is not None:
        _filename_fns[media_type][key] = filename_fn


def get_producer(
    media_type: MediaType, identity: "type[ObjectId]", level: str,
    implementation: str = PIXINSIGHT,
) -> Optional[Callable[..., Path]]:
    """Legacy: look up a producer by (identity, level, implementation).
    New code: import the producing op directly."""
    return _registry[media_type].get((identity, level, implementation))


def get_filename_fn(
    media_type: MediaType, identity: "type[ObjectId]", level: str,
    implementation: str = PIXINSIGHT,
) -> Optional[Callable[..., Path]]:
    """Legacy: look up a producer's resolved filename_fn. New code:
    `op.filename_fn` or `op.filename_fns[(identity, level)]`."""
    return _filename_fns[media_type].get((identity, level, implementation))


def _mark_valid(identity_cls: type, level: str, implementation: str, args: tuple) -> None:
    """Mark a product as valid after its producer completes."""
    # The first argument to a producer is always the primary object
    # (Stack, Mosaic, etc.) which has stable_hash.
    obj = args[0] if args else None
    if obj is None or not hasattr(obj, "stable_hash"):
        return
    key = (identity_cls.object_type.__name__.lower(), obj.stable_hash, str(level), str(implementation))
    from context import context
    context.or_equals("validity", {key: True})


def invalidate_object(object_type: str, object_id: str, implementation: str = None) -> None:
    """Mark all quality levels of an object as invalid."""
    from context import context
    implementations = [implementation] if implementation else ALL_IMPLEMENTATIONS
    updates = {}
    for level in ALL_QUALITY_LEVELS:
        for impl in implementations:
            updates[(object_type, object_id, str(level), impl)] = False
    context.or_equals("validity", updates)


def is_object_valid(
    object_type: str, object_id: str, level: str,
    implementation: str = "PixInsight",
) -> bool:
    """Query validity, deriving parent validity from children.

    Returns True if the product is up to date, False otherwise
    (stale or never produced). `implementation` defaults to the
    PixInsight backend — there's no "user-active implementation"
    setting anymore; callers that care about a specific backend
    pass it explicitly.
    """
    from context import context

    if object_type == "mosaic":
        mosaics = context.mosaics or {}
        if object_id not in mosaics:
            return False
        mosaic = mosaics[object_id].get()
        return all(is_object_valid("target", tid, level, implementation) for tid in mosaic.target_ids)

    if object_type == "target":
        targets = context.targets or {}
        if object_id not in targets:
            return False
        target = targets[object_id].get()
        return all(is_object_valid("stack", sid, level, implementation) for sid in target.stack_ids)

    # Leaf: valid only if explicitly marked True.
    validity = context.get("validity", {})
    return validity.get((object_type, object_id, level, implementation)) is True


def fully_resolve(
    identity_cls: type, level: str, filename_fn: FilenameFn,
    implementation: str = PIXINSIGHT, op_name: str | None = None,
    multi_mode: bool = False,
):
    """Resolve a filename_fn against an output directory layout keyed
    by op_name (the producer's `__name__`). The level/impl axes that
    used to live in the directory hierarchy now collapse into the
    op_name layer — `Stack/Master/PixInsight/light_*.xisf` becomes
    `Stack/stack_files/light_*.xisf` — because after the pipeline
    collapse, "what op wrote this" replaces "which quality + which
    backend" as the artifact's primary identity.

    Multi-mode ops (those whose @makes_image registers under multiple
    (level, impl) combos with the same function — `place_image`,
    `merge_images_xisf`, `merge_target_channel`, etc.) inject the
    (level, impl) tuple into the filename STEM to avoid collisions
    within their shared op_name directory.
    """
    object_type_name = identity_cls.object_type.__name__
    # Backwards-compat: fall back to legacy (level, impl) layout if
    # @makes_image hasn't passed op_name explicitly. Won't trigger
    # for any in-tree producer because _bind_producer always passes
    # op_name; here so external callers (none today) don't break.
    if op_name is None:
        op_name = f"{level}_{implementation}"

    def inner(*args, **kwargs):
        filename = filename_fn(*args, **kwargs)
        assert isinstance(filename, ContentHashable)
        output_dir = context.output_dir
        assert output_dir is not None, (
            "context.output_dir is unset on this worker — "
            "state_machine._populate_context must run before any "
            "filename can be fully resolved."
        )
        # Selection-style filename_fns (`get_best_light_frame`,
        # `get_first_frame`) return ABSOLUTE paths to existing raw
        # frames; mangling those into the output-dir scheme would
        # invent paths that don't exist on disk. Detect and pass
        # through unchanged.
        path_str = os.fspath(filename)
        if os.path.isabs(path_str):
            return filename
        if multi_mode:
            # Inject `_<level>_<impl>` into the stem so the multiple
            # (level, impl) registrations of the same op_name don't
            # clobber each other inside the shared op directory.
            suffix = filename.suffix
            new_name = filename.stem + f"_{level}_{implementation}" + suffix
            filename = type(filename)(new_name)
        return filename.resolve_at(output_dir / object_type_name / op_name)

    return inner


def _extract_identity(
    identity_cls: type[ObjectId],
    func: Callable,
    args: tuple,
    kwargs: dict,
) -> ObjectId:
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    id_fields = {f.name for f in dataclasses.fields(identity_cls)}
    id_kwargs = {name: bound.arguments[name] for name in id_fields}
    return identity_cls(**id_kwargs)


def _detect_level(
    func: Callable, args: tuple, kwargs: dict, levels: list[str],
) -> list[str]:
    if len(levels) == 1:
        return levels
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    if 'mode' in bound.arguments:
        mode = bound.arguments['mode']
        if isinstance(mode, str):
            return [mode]
    return levels


def _generate_presentations(
    func: Callable,
    result,
    args: tuple,
    kwargs: dict,
    pres_entries: list[dict],
) -> None:
    if not pres_entries:
        return

    for entry in pres_entries:
        identity_cls = entry['identity_cls']
        presentations = entry['presentations']
        implementations = entry['implementations']
        levels = entry['levels']

        try:
            obj_id = _extract_identity(identity_cls, func, args, kwargs)
        except (TypeError, KeyError) as e:
            print(f"### Failed to extract {identity_cls.__name__}: {e}")
            continue

        active_levels = _detect_level(func, args, kwargs, levels)

        for presentation in presentations:
            if presentation == PresentationType.THUMBNAIL:
                # Thumbnails are fully lazy now. The web server rebuilds
                # them on first view into the bounded evictable cache:
                # file-backed producers from their on-disk master/frame,
                # compute-backed ones (flat-correction Run thumbnails,
                # no filename_fn) by invoking the cheap producer on miss
                # (see web_server/backend/thumbnails._regenerate_
                # thumbnail). Eager generation here only refilled the
                # ~95 GB and stalled the cold start, so it is gone.
                continue

            elif presentation == PresentationType.TILE_PYRAMID:
                tile_path = _make_tile_pyramid(result)
                if tile_path is None:
                    continue
                # Tile-pyramid identity is "this op's output for this
                # object" — the (impl, level) axes the legacy registry
                # used are folded into op_name (e.g. `stack_files` vs
                # `downscale_master`). One entry per @makes_image
                # decoration, regardless of how many (impl, level) pairs
                # the legacy registration declared.
                op_name = func.__name__
                key = obj_id.presentation_cache_key(
                    PresentationType.TILE_PYRAMID, op_name,
                )
                context[key] = str(tile_path)
                print(f"### Stored tile_pyramid {key}")


def _make_thumbnail(result, resize_array, autostretch, array_to_png, width=2048):
    if isinstance(result, np.ndarray):
        return array_to_png(
            autostretch(np.clip(resize_array(result, width), 0.0, 1.0))
        )
    if isinstance(result, tempfile._TemporaryFileWrapper):
        filename = result.name
    else:
        filename = os.fspath(result)
    if os.path.exists(filename):
        from file_io.generic import read_image
        return array_to_png(
            autostretch(np.clip(resize_array(read_image(filename), width), 0.0, 1.0))
        )
    return None


def _make_tile_pyramid(result) -> "Path | None":
    if isinstance(result, np.ndarray):
        return None
    if isinstance(result, tempfile._TemporaryFileWrapper):
        source = Path(result.name)
    else:
        source = Path(os.fspath(result))
    if not source.exists():
        return None
    if source.suffix.lower() in (".tif", ".tiff"):
        return source
    tile_path = source.with_suffix(".tiff")
    try:
        from previews.create_tile_pyramid import default_make_tile_pyramid
        default_make_tile_pyramid(source, tile_path)
    except Exception as e:
        print(f"### Failed to make tile pyramid for {source}: {e}")
        return None
    return tile_path


_FILE_TYPE_SUFFIX = {
    'FitsFilePath': '.fits',
    'XisfFilePath': '.xisf',
    'TiffFilePath': '.tiff',
}


def _make_filename_fn(identity_cls: type[ObjectId], file_type: type):
    """Generate a filename_fn from an ObjectId class with filename_stem()."""
    suffix = _FILE_TYPE_SUFFIX.get(file_type.__name__, '.fits')
    fields = [f.name for f in dataclasses.fields(identity_cls)]
    def fn(*args, **kwargs):
        id_kwargs = {}
        for i, name in enumerate(fields):
            if name in kwargs:
                id_kwargs[name] = kwargs[name]
            elif i < len(args):
                id_kwargs[name] = args[i]
        return file_type(identity_cls(**id_kwargs).filename_stem() + suffix)
    return fn


def _make_dispatch_filename_fn(func: Callable):
    """Convenience `producer.filename_fn` that resolves across every
    registered `(identity, level)` filename_fn. Monomorphic producers
    have exactly one and behave identically; polymorphic ones
    (stack_files registers StackId AND CalibrationStackId) dispatch on
    the one whose filename_stem() applies. The identities' filename_stem
    predicates are disjoint by image_type and signal "not mine" with a
    ValueError — catch ONLY that, so a genuine TypeError/KeyError in a
    filename_fn stays loud instead of being silently masked (the
    swallow-and-try-next pattern that hid the original 404 bug)."""
    def dispatch(*args, **kwargs):
        last_exc = None
        for fn in func.filename_fns.values():
            try:
                return fn(*args, **kwargs)
            except ValueError as e:
                last_exc = e
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("no filename_fn registered for producer")
    return dispatch


def _chain_compute_image_metadata(handle):
    """Trace-time hook: record a compute_image_metadata node downstream
    of `handle` and return its handle. Attached to every @makes_image-
    decorated op so user pipelines don't need to write the
    `compute_image_metadata(stack(...))` pattern explicitly. Invoked by
    `cached_decorator`'s trace branch through the generic `_trace_hooks`
    list."""
    from dependency_graph.tracing import record
    from metadata.image_metadata import compute_image_metadata
    return record(compute_image_metadata, handle)


def _bind_producer(
    media_type: MediaType,
    implementation: str | list[str],
    levels: list[str],
    filename_fn: FilenameFn | None,
    identity: "type[ObjectId]",
    presentations: "list[PresentationType] | None" = None,
    file_type: "type | None" = None,
    display: str | None = None,
):
    """Decorator that registers a function as a producer for the given
    (identity, level, implementation) combinations. The produced
    object_type is derived from ``identity.object_type``.

    `display` (optional): free-form label naming this op as a user-
    visible tap point within a pipeline. When a pipeline traces, every
    @makes_image Node whose task has `_display` set is recorded in the
    PipelineOutputIndex as a tap point under that name. Skip `display`
    for intermediate ops (e.g. per-strip merge_images) that shouldn't
    surface in the dashboard.

    Usage::

        @node(...)
        @makes_image(PIXINSIGHT, [MASTER], identity=StackId,
                     display="Master")
        def stack_files(stack, config): ...
    """
    assert identity is not None, "@makes_image/@makes_video requires identity="
    implementations = implementation if isinstance(implementation, list) else [implementation]

    # Auto-generate filename_fn from identity when file_type is provided.
    if filename_fn is None and identity is not None and file_type is not None:
        if hasattr(identity, 'filename_stem'):
            filename_fn = _make_filename_fn(identity, file_type)

    def decorator(func: Callable[..., Path]):
        # Decoration order matters: @makes_image / @makes_video must
        # be applied *inside* @node so that we wrap the raw function
        # (not the Ray-task wrapper). cached_decorator stamps
        # `_is_task = True` on the wrapper it returns, so spotting
        # that attribute on `func` means the user wrote the decorators
        # in the wrong order.
        if getattr(func, '_is_task', False):
            deco = "makes_image" if media_type == MediaType.IMAGE else "makes_video"
            raise TypeError(
                f"@{deco} must be applied inside @node, not outside. "
                f"Got @{deco}(identity={identity.__name__}) wrapping "
                f"a Ray task. Correct order:\n"
                f"    @node(...)\n"
                f"    @{deco}(...)\n"
                f"    def {getattr(func, '__name__', '...')}(...): ..."
            )

        # Validate identity fields match producer params at decoration time.
        import dataclasses
        import inspect
        id_fields = {f.name for f in dataclasses.fields(identity)}
        raw_func = func._original if hasattr(func, '_original') else func
        raw_func = raw_func.__wrapped__ if hasattr(raw_func, '__wrapped__') else raw_func
        producer_params = set(inspect.signature(raw_func).parameters.keys())
        missing = id_fields - producer_params
        if missing:
            raise TypeError(
                f"{identity.__name__} has fields {missing} that don't match "
                f"any parameter of {getattr(func, '__name__', '...')}. "
                f"ObjectId field names must be a subset of producer parameter names."
            )

        # Accumulate validity entries (shared across stacked decorators
        # for polymorphic producers like stack_files).
        if not hasattr(func, '_validity_entries'):
            func._validity_entries = []

        # Track registry keys so @node can update the registry to
        # point at the outer (Ray-task) wrapper.
        if not hasattr(func, '_registry_keys'):
            func._registry_keys = []

        # Accumulate presentation entries for ObjectId-based producers.
        if not hasattr(func, '_presentation_entries'):
            func._presentation_entries = []

        op_name = func.__name__
        # Ops whose @makes_image declares multiple (impl, level) combos
        # share a single op_name directory, so per-combo filenames need
        # the (level, impl) discriminator baked into the stem.
        multi_mode = len(implementations) * len(levels) > 1

        for impl in implementations:
            for level in levels:
                resolved = None
                if filename_fn is not None:
                    resolved = fully_resolve(
                        identity, level, filename_fn, impl,
                        op_name=op_name, multi_mode=multi_mode,
                    )
                    if not hasattr(func, 'filename_fns'):
                        func.filename_fns = {}
                    func.filename_fns[(identity, level)] = resolved
                    # Convenience alias. For polymorphic producers
                    # (stack_files: StackId + CalibrationStackId, whose
                    # filename_stem()s ValueError on a mismatched
                    # image_type) a single "last wins" fn would crash on
                    # half the inputs, so dispatch across all registered
                    # filename_fns and use the first that resolves.
                    func.filename_fn = _make_dispatch_filename_fn(func)

                func._validity_entries.append((identity, level, impl))
                func._registry_keys.append((media_type, identity, level, impl))

        if presentations:
            func._presentation_entries.append({
                'identity_cls': identity,
                'presentations': list(presentations),
                'implementations': implementations,
                'levels': levels,
            })

        # Wrap the function once. Subsequent decorator calls (stacked
        # @makes_image decorators for polymorphic producers) skip the
        # wrap and just append to the closure-captured `entries` list.
        if getattr(func, '_makes_image_wrapped', False):
            wrapped = func
        else:
            from functools import wraps
            from file_io.sidecar import discard_sidecar
            inner_func = func
            entries = func._validity_entries  # closure-captured (shared)
            pres_entries = func._presentation_entries  # closure-captured (shared)

            @wraps(inner_func)
            def wrapped(*args, **kwargs):
                result = inner_func(*args, **kwargs)
                if result is not None:
                    for idcls, lvl, impl in entries:
                        _mark_valid(idcls, lvl, impl, args)
                    if not isinstance(result, np.ndarray):
                        discard_sidecar(result)
                    _generate_presentations(
                        inner_func, result, args, kwargs, pres_entries,
                    )
                return result

            wrapped._validity_entries = entries
            wrapped._presentation_entries = pres_entries
            wrapped._makes_image_wrapped = True
            wrapped._registry_keys = func._registry_keys
            # Trace-time hook: when this op is called inside a
            # @graph_builder / @pipeline trace, automatically record a
            # compute_image_metadata node downstream. Every @makes_image
            # op produces a file on disk whose canonical downstream
            # consumer is its ImageMetadata; auto-chaining lets user
            # pipelines skip the boilerplate
            # `compute_image_metadata(stack(...))` pattern. The hook is
            # invoked by cached_decorator's trace branch, which knows
            # only about the generic hook list — not about
            # compute_image_metadata specifically.
            wrapped._trace_hooks = [_chain_compute_image_metadata]
            if hasattr(inner_func, 'filename_fns'):
                wrapped.filename_fns = inner_func.filename_fns
            if hasattr(inner_func, 'filename_fn'):
                wrapped.filename_fn = inner_func.filename_fn
            # `_display` is set on the wrapped fn so the trace-time
            # tap-point collector can identify which Nodes the
            # pipeline author marked as user-visible. None means
            # "intermediate" (not a tap point).
            if display is not None:
                wrapped._display = display
            if hasattr(inner_func, '_display'):
                wrapped._display = inner_func._display

        # Populate the legacy registry so the view-side endpoints
        # that still call get_producer / get_filename_fn keep working.
        # In-tree internal ops don't read these dicts; they import
        # the producing op directly and use `op.filename_fn(...)`.
        for impl in implementations:
            for level in levels:
                resolved = None
                if filename_fn is not None:
                    resolved = fully_resolve(
                        identity, level, filename_fn, impl,
                        op_name=op_name, multi_mode=multi_mode,
                    )
                register_producer(
                    media_type, level, wrapped,
                    filename_fn=resolved,
                    implementation=impl,
                    identity=identity,
                )

        return wrapped

    return decorator


def makes_image(
    implementation: str,
    levels: list[str], filename_fn: FilenameFn | None = None,
    identity: type[ObjectId] = None,
    presentations: list[PresentationType] | None = None,
    file_type: type | None = None,
    display: str | None = None,
):
    return _bind_producer(
        MediaType.IMAGE, implementation, levels, filename_fn,
        identity=identity, presentations=presentations, file_type=file_type,
        display=display,
    )


def makes_video(
    implementation: str,
    levels: list[str], filename_fn: FilenameFn | None = None,
    identity: type[ObjectId] = None,
    presentations: list[PresentationType] | None = None,
    file_type: type | None = None,
    display: str | None = None,
):
    return _bind_producer(
        MediaType.VIDEO, implementation, levels, filename_fn,
        identity=identity, presentations=presentations, file_type=file_type,
        display=display,
    )
