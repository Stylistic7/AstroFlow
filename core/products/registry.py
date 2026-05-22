import dataclasses
import inspect
import os
import tempfile
import typing
from enum import Enum
from pathlib import Path
from typing import Callable, Union

import numpy as np

from file_io.types import ContentHashable
from products.identity import ObjectId, PresentationType
from context import context


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


type FilenameFn = Union[str, Callable[..., Path]]


# Per-identity producer list. @makes_image / @makes_video appends to
# `_producers[media_type][identity]` so view-side endpoints that need
# "every op that writes a StackId-shaped image" (the bare-thumbnail
# regen path) can enumerate them. Order is decoration order.
_producers: dict[MediaType, dict[type, list[Callable]]] = {
    MediaType.IMAGE: {},
    MediaType.VIDEO: {},
}


_producers_imported = False


def _ensure_producers_imported():
    """Import every module that calls @makes_image / @makes_video so
    their decoration side-effects run before any producers_for()
    lookup. Idempotent; pays the import cost exactly once."""
    global _producers_imported
    if _producers_imported:
        return
    _producers_imported = True
    from products import ensure_producers_registered
    ensure_producers_registered()


def producers_for(
    identity: type[ObjectId], media_type: MediaType = MediaType.IMAGE,
) -> list[Callable]:
    """Every op registered to produce `identity`-shaped artifacts of
    `media_type`. Returns the @makes_image-decorated functions in
    decoration order. Polymorphic ops (stack_files registered for
    both StackId and CalibrationStackId) appear in each identity's
    list."""
    _ensure_producers_imported()
    return list(_producers[media_type].get(identity, []))


def fully_resolve(
    identity_cls: type, filename_fn: FilenameFn, op_name: str,
    *,
    level: str | None = None,
    implementation: str | None = None,
):
    """Resolve a filename_fn against the output-directory layout:

        <output_dir>/<object_type>/<op_name>/<identity_stem>.<ext>

    `op_name` (the producer's `__name__`) is the entire on-disk
    discriminator. Different ops writing the same identity-shape
    land in different directories — pipelines that swap one op for
    another (`stack_files` vs `stack_files_downscaled`) never
    collide.

    `level`/`implementation` salt the filename stem for ops whose
    single function body registers under multiple (impl, level)
    combos (place_image, merge_images, merge_target_channel,
    store_mosaic_color_channel_xisf). Single-mode ops pass None
    for both and get the clean `<identity_stem>.<ext>` filename.
    """
    object_type_name = identity_cls.object_type.__name__
    multi_mode = level is not None and implementation is not None

    def inner(*args, **kwargs):
        filename = filename_fn(*args, **kwargs)
        assert isinstance(filename, ContentHashable)
        output_dir = context.output_dir
        assert output_dir is not None, (
            "context.output_dir is unset on this worker — "
            "state_machine._populate_context must run before any "
            "filename can be fully resolved."
        )
        # Selection-style filename_fns return absolute paths to
        # existing raw frames; mangling those would invent paths
        # that don't exist. Pass through unchanged.
        path_str = os.fspath(filename)
        if os.path.isabs(path_str):
            return filename
        if multi_mode:
            suffix = filename.suffix
            new_name = filename.stem + f"_{level}_{implementation}" + suffix
            filename = type(filename)(new_name)
        return filename.resolve_at(output_dir / object_type_name / op_name)

    return inner


def _extract_identity(
    identity_cls: type[ObjectId],
    func: Callable, args: tuple, kwargs: dict,
) -> ObjectId:
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    id_fields = {f.name for f in dataclasses.fields(identity_cls)}
    id_kwargs = {name: bound.arguments[name] for name in id_fields}
    return identity_cls(**id_kwargs)


_FILE_TYPE_SUFFIX = {
    'FitsFilePath': '.fits',
    'XisfFilePath': '.xisf',
    'TiffFilePath': '.tiff',
}


def _infer_file_type(func: Callable):
    """Inspect the producer's return annotation to pick the on-disk
    file type. Supports:

      def f(...) -> XisfFilePath: ...
      def f(...) -> WithSidecar[XisfFilePath]: ...

    Returns None for ops without an annotated file type (`ndarray`
    returns, ungenericised `WithSidecar`, etc.) — those ops don't get
    an auto-generated filename_fn and must supply one explicitly."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return None
    ret = sig.return_annotation
    if ret is inspect.Signature.empty:
        return None
    # Direct file-type annotation.
    if getattr(ret, '__name__', None) in _FILE_TYPE_SUFFIX:
        return ret
    # Generic wrapper (e.g. `WithSidecar[XisfFilePath]`): unwrap to
    # the parameter.
    args = typing.get_args(ret)
    for a in args:
        if getattr(a, '__name__', None) in _FILE_TYPE_SUFFIX:
            return a
    return None


def _make_filename_fn(identity_cls: type[ObjectId], file_type: type):
    """Generate a filename_fn from an ObjectId class with filename_stem()."""
    suffix = _FILE_TYPE_SUFFIX.get(file_type.__name__)
    if suffix is None:
        return None
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
    """`producer.filename_fn` for polymorphic producers (stack_files
    registers both StackId and CalibrationStackId). Tries each
    registered identity's filename_fn and returns the first that
    resolves; identity-mismatched callees raise ValueError, anything
    else propagates loud — masking a real TypeError/KeyError once hid
    a 404 bug for weeks."""
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


# Auto-generated tile pyramids — anything large enough that scrolling
# at full resolution would be sluggish gets one. ~8 megapixels is the
# 4K-display threshold; below that an autostretch + array_to_png
# thumbnail covers all the cases we care about.
_TILE_PYRAMID_MIN_PIXELS = 8 * 1024 * 1024


def _make_tile_pyramid_if_warranted(result) -> "Path | None":
    """Build a tile pyramid sibling for `result` iff the image is
    large enough to justify one. ndarray results are returned by
    cheap subframe-preview ops — already small, no pyramid needed.
    File-based results get a `.tiff` sibling when the file's image
    dimensions exceed 8 megapixels."""
    if isinstance(result, np.ndarray):
        return None
    if isinstance(result, tempfile._TemporaryFileWrapper):
        source = Path(result.name)
    else:
        source = Path(os.fspath(result))
    if not source.exists():
        return None
    # Skip generation if the file itself is already a tiff (it's the
    # tile pyramid).
    if source.suffix.lower() in (".tif", ".tiff"):
        return source
    # Cheap size check — header-only read.
    try:
        from file_io.generic import read_image_geometry
        geom = read_image_geometry(source)
        pixels = geom.width_pixels * geom.height_pixels
    except Exception:
        pixels = None
    if pixels is not None and pixels < _TILE_PYRAMID_MIN_PIXELS:
        return None
    tile_path = source.with_suffix(".tiff")
    try:
        from previews.create_tile_pyramid import default_make_tile_pyramid
        default_make_tile_pyramid(source, tile_path)
    except Exception as e:
        print(f"### Failed to make tile pyramid for {source}: {e}")
        return None
    return tile_path


def _make_thumbnail(result, resize_array, autostretch, array_to_png, width=3840):
    """Build a 4K-wide PNG thumbnail from `result`. `width=3840`
    matches a 4K display so the thumbnail itself looks good at full
    resolution; the on-disk image is downsampled to that width and
    autostretched. Result types: an ndarray (returned directly), a
    tempfile, or a Path-like on-disk image file."""
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


def _generate_presentations(
    func: Callable, result, args: tuple, kwargs: dict,
    identities: list[type],
) -> None:
    """Auto-generate sibling artifacts: a tile pyramid TIFF next to
    the master output (when the image is large enough to justify
    one). The web server's bare-thumbnail route lazy-generates the
    thumbnail PNG on miss from either the pyramid or the master,
    so we don't pre-build it here.

    Iterated over each (identity) the @makes_image stack contributed
    — for polymorphic ops the same result is keyed under each
    identity it can be looked up by."""
    if result is None:
        return
    tile_path = _make_tile_pyramid_if_warranted(result)
    if tile_path is None:
        return
    op_name = func.__name__
    for identity_cls in identities:
        try:
            obj_id = _extract_identity(identity_cls, func, args, kwargs)
        except (TypeError, KeyError):
            continue
        key = obj_id.presentation_cache_key(
            PresentationType.TILE_PYRAMID, op_name,
        )
        context[key] = str(tile_path)


def _chain_compute_image_metadata(handle):
    """Trace-time hook: every @makes_image-decorated op produces a
    file whose canonical downstream consumer is its ImageMetadata,
    so the wrapper records a `compute_image_metadata` node downstream
    automatically. User pipelines skip the
    `compute_image_metadata(stack(...))` boilerplate. Invoked by
    `cached_decorator`'s trace branch through `_trace_hooks`."""
    from dependency_graph.tracing import record
    from metadata.image_metadata import compute_image_metadata
    return record(compute_image_metadata, handle)


def _bind_producer(
    media_type: MediaType,
    implementation: "str | list[str] | None",
    levels: "list[str] | None",
    filename_fn: FilenameFn | None,
    identity: "type[ObjectId]",
    display: str | None = None,
):
    """Register a function as the producer of `identity`-shaped
    artifacts. The on-disk path is keyed by the producer's
    `__name__`:

        <output_dir>/<object_type>/<op_name>/<identity_stem>.<ext>

    File type is inferred from the function's return annotation:
    `-> XisfFilePath` or `-> WithSidecar[XisfFilePath]` etc. Ops
    returning ndarray have no on-disk path; they must supply an
    explicit `filename_fn` if they need one (the subframe-thumbnail
    ops use this).

    `display` (optional): user-visible tap-point label. The
    pipeline-trace tap collector reads `_display` and records every
    tagged Node in the PipelineOutputIndex; the dashboard surfaces
    those entries per-pipeline. Skip for intermediate ops
    (per-strip merges, per-panel placements) that shouldn't appear
    as tap points.

    Usage::

        @node(...)
        @makes_image(identity=StackId, display="Master")
        def stack_files(stack, config) -> WithSidecar[XisfFilePath]:
            ...

    `implementation` and `levels` are still accepted by ops whose
    single function body produces multiple on-disk variants
    discriminated by a `mode` arg (place_image, merge_images,
    merge_target_channel, store_mosaic_color_channel_xisf). The
    (impl, level) tuple salts the filename stem so the variants
    don't clobber each other inside the shared op-name directory.
    Splitting these into one op per variant is the long-term
    direction.
    """
    assert identity is not None, "@makes_image/@makes_video requires identity="

    if implementation is None:
        implementations = [None]
    elif isinstance(implementation, list):
        implementations = implementation
    else:
        implementations = [implementation]
    if levels is None:
        levels = [None]

    def decorator(func: Callable[..., Path]):
        # @makes_image must wrap the raw function, not the Ray-task
        # wrapper. cached_decorator sets `_is_task = True` on its
        # output — spotting it on `func` means the decorators are
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

        # Auto-generate filename_fn from the return annotation when
        # no explicit one was given.
        local_filename_fn = filename_fn
        if local_filename_fn is None and hasattr(identity, 'filename_stem'):
            file_type = _infer_file_type(raw_func)
            if file_type is not None:
                local_filename_fn = _make_filename_fn(identity, file_type)

        if not hasattr(func, '_identities'):
            func._identities = []
        func._identities.append(identity)

        op_name = func.__name__
        # Multi-mode ops register one body under N (impl, level) combos
        # and need an in-stem discriminator to keep the variants apart.
        multi_mode = len(implementations) * len(levels) > 1

        for impl in implementations:
            for level in levels:
                if local_filename_fn is None:
                    continue
                fn_level = level if multi_mode else None
                fn_impl = impl if multi_mode else None
                resolved = fully_resolve(
                    identity, local_filename_fn, op_name,
                    level=fn_level, implementation=fn_impl,
                )
                if not hasattr(func, 'filename_fns'):
                    func.filename_fns = {}
                func.filename_fns[(identity, level)] = resolved
                func.filename_fn = _make_dispatch_filename_fn(func)

        # Wrap once; stacked decorators (polymorphic identities) share
        # the closure-captured `identities` list and skip re-wrapping.
        if getattr(func, '_makes_image_wrapped', False):
            wrapped = func
        else:
            from functools import wraps
            from file_io.sidecar import discard_sidecar
            inner_func = func
            identities = func._identities

            @wraps(inner_func)
            def wrapped(*args, **kwargs):
                result = inner_func(*args, **kwargs)
                if result is not None:
                    if not isinstance(result, np.ndarray):
                        discard_sidecar(result)
                    _generate_presentations(
                        inner_func, result, args, kwargs, identities,
                    )
                return result

            wrapped._identities = identities
            wrapped._makes_image_wrapped = True
            # Trace-time hook: when this op is called inside a
            # @pipeline trace, auto-chain a compute_image_metadata
            # node downstream so user pipelines don't have to write
            # the boilerplate themselves.
            wrapped._trace_hooks = [_chain_compute_image_metadata]
            if hasattr(inner_func, 'filename_fns'):
                wrapped.filename_fns = inner_func.filename_fns
            if hasattr(inner_func, 'filename_fn'):
                wrapped.filename_fn = inner_func.filename_fn
            if display is not None:
                wrapped._display = display
            if hasattr(inner_func, '_display'):
                wrapped._display = inner_func._display

        # Register the wrapped op for view-side enumeration.
        _producers[media_type].setdefault(identity, []).append(wrapped)

        return wrapped

    return decorator


def makes_image(
    implementation=None,
    levels=None,
    filename_fn: FilenameFn | None = None,
    identity: type[ObjectId] = None,
    display: str | None = None,
):
    """Decorate an op as the producer of `identity`-shaped images.

    Output goes to
    `<output_dir>/<object_type>/<op_name>/<identity_stem>.<ext>`.
    The extension comes from the function's return-type annotation
    (`-> WithSidecar[XisfFilePath]` etc.).

    Tile pyramids are auto-generated next to large outputs (over 8
    megapixels — the 4K display threshold). The web server's bare-
    thumbnail route lazy-generates a 4K-wide PNG on miss.

    Single-mode (most ops)::

        @node(...)
        @makes_image(identity=StackId, display="Master")
        def stack_files(stack, config) -> WithSidecar[XisfFilePath]:
            ...

    Multi-mode (place_image, merge_images, merge_target_channel,
    store_mosaic_color_channel_xisf) accept positional
    `(implementation, levels)` and inject `_<level>_<impl>` into the
    filename stem so a single function body registered under
    multiple (impl, level) combos doesn't clobber its own outputs.
    Splitting these into one op per variant is the cleaner future
    state.
    """
    return _bind_producer(
        MediaType.IMAGE, implementation, levels, filename_fn,
        identity=identity, display=display,
    )


def makes_video(
    implementation=None,
    levels=None,
    filename_fn: FilenameFn | None = None,
    identity: type[ObjectId] = None,
    display: str | None = None,
):
    """Video counterpart of @makes_image. Same signature; produces a
    video artifact at
    `<output_dir>/<object_type>/<op_name>/<identity_stem>.<ext>`.
    """
    return _bind_producer(
        MediaType.VIDEO, implementation, levels, filename_fn,
        identity=identity, display=display,
    )
