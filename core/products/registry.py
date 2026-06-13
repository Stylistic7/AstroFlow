"""Image-lifecycle and producer-registry helpers.

Public surface — all consumed by `@node` (decorators/decorators.py),
which is the sole orchestrator that calls these in the right order
with the right shared artifacts.

Helpers:

  * `build_filename_fn(func, canonical_call_hash)` — construct the
    resolved `filename_fn` for an artifact-producing op. Returns one
    function (no polymorphic dispatcher — every @node has exactly one
    routing path).
  * `makes_image_wrap(func, *, canonical_call_hash)` — wrap `func` so
    that after the function returns, the result file's `content_hash`
    header is stamped to the call_hash and presentation artifacts
    (tile pyramid sibling) are generated.
  * `register_producer(wrapper, func)` — add the final wrapper to
    `_producers[type_of_first_param]` for view-side enumeration.

Also kept here: the producer registry itself (`_producers`,
`producers_for`), file-type inference utilities, and the tile pyramid
generator.
"""
import inspect
import os
import tempfile
import typing
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np

from file_io.types import ContentHashable, WithSidecar
from context import context


type FilenameFn = Union[str, Callable[..., Path]]


# ---------------------------------------------------------------------
# Producer registry
# ---------------------------------------------------------------------

# `_producers[arg_type]` — every artifact-producing `@node` appends
# itself, keyed by the type of its first parameter, so view-side
# endpoints can enumerate "every op whose first arg is a Stack /
# Mosaic / etc."
_producers: dict[type, list[Callable]] = {}

_producers_imported = False


def _ensure_producers_imported():
    """Trigger imports of every module containing `@node(identity=...)`
    so decoration side-effects (which populate `_producers`) run before
    any `producers_for()` lookup. Idempotent; pays the import cost once."""
    global _producers_imported
    if _producers_imported:
        return
    _producers_imported = True
    from products import ensure_producers_registered
    ensure_producers_registered()


def producers_for(arg_type: type) -> list[Callable]:
    """Every artifact-producing op whose first argument can accept an
    instance of `arg_type`. Returns the wrappers in registration order.

    Walks `arg_type.__mro__` so an op annotated with a base class
    (e.g. `stack: BaseStack`) surfaces for `producers_for(Stack)` AND
    `producers_for(CalibrationStack)` — both are subclasses of
    `BaseStack`, so passing either as the first arg works at runtime.
    """
    _ensure_producers_imported()
    out: list[Callable] = []
    for cls in arg_type.__mro__:
        out.extend(_producers.get(cls, []))
    return out


def register_producer(wrapper: Callable, func: Callable) -> None:
    """Register `wrapper` under the first-parameter type of `func` so
    `producers_for(arg_type)` can later find it.

    The key is inferred from the original function's first-parameter
    annotation — that's the "anchor type" the artifact belongs to.
    """
    arg_type = _first_param_type(func)
    if arg_type is None:
        return
    _producers.setdefault(arg_type, []).append(wrapper)


def _first_param_type(func: Callable) -> type | None:
    """Return the annotated type of the first positional parameter, or
    None if there's no annotation. The decorator may have wrapped the
    function; peel back to `__wrapped__` if present so we read the
    user's annotations, not the wrapper's `*args` signature."""
    raw = func.__wrapped__ if hasattr(func, '__wrapped__') else func
    try:
        sig = inspect.signature(raw)
    except (TypeError, ValueError):
        return None
    params = list(sig.parameters.values())
    if not params:
        return None
    ann = params[0].annotation
    if ann is inspect.Parameter.empty or not isinstance(ann, type):
        return None
    return ann


def image_producer(*, display: str | None = None, final: bool = False):
    """Lightweight `@image_producer` for non-cached, non-Ray functions
    that still need to be discoverable via `producers_for(type_of_first_arg)`.
    Routing keys off the first-parameter type.
    """
    def decorator(func):
        if display is not None:
            func._display = display
        if final:
            func._final = True
        register_producer(func, func)
        return func
    return decorator


# ---------------------------------------------------------------------
# File-type inference + path resolution
# ---------------------------------------------------------------------

_FILE_TYPE_SUFFIX = {
    'FitsFilePath': '.fits',
    'XisfFilePath': '.xisf',
    'TiffFilePath': '.tiff',
}


def _infer_file_type(func: Callable):
    """Inspect the producer's return annotation to pick the on-disk file
    type. Supports `-> XisfFilePath` and `-> WithSidecar[XisfFilePath]`.

    Returns None for ops without an annotated file type (`ndarray` returns,
    ungenericised `WithSidecar`, etc.) — they don't get an auto-generated
    filename_fn."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return None
    ret = sig.return_annotation
    if ret is inspect.Signature.empty:
        return None
    if getattr(ret, '__name__', None) in _FILE_TYPE_SUFFIX:
        return ret
    for a in typing.get_args(ret):
        if getattr(a, '__name__', None) in _FILE_TYPE_SUFFIX:
            return a
    return None


def fully_resolve(filename_fn: FilenameFn, op_name: str):
    """Wrap `filename_fn` so its output is prefixed with the output-dir
    routing path:

        <output_dir>/<op_name>/<filename_fn(...)>

    The op's `__name__` is the entire organizing principle on disk:
    the function's type signature is fixed, so the op name uniquely
    determines what's stored. Content-hashed filenames make
    collisions impossible across ops. Selection-style absolute paths
    are passed through unchanged (used for raw-frame producers).
    """
    def inner(*args, **kwargs):
        filename = filename_fn(*args, **kwargs)
        assert isinstance(filename, ContentHashable)
        output_dir = context.output_dir
        assert output_dir is not None, (
            "context.output_dir is unset on this worker — "
            "state_machine._populate_context must run before any "
            "filename can be fully resolved."
        )
        path_str = os.fspath(filename)
        if os.path.isabs(path_str):
            return filename
        return filename.resolve_at(output_dir / op_name)

    return inner


def build_filename_fn(
    func: Callable,
    canonical_call_hash: Callable,
) -> Callable | None:
    """Construct the resolved `filename_fn` for an artifact-producing
    op. Returns the single function (no polymorphic dispatcher — every
    @node has exactly one routing path now).

    The filename_fn:
      1. Calls `canonical_call_hash(*args, **kwargs)` for THIS call.
      2. Returns `<call_hash>.<ext>` wrapped by `fully_resolve`, which
         prepends `<output_dir>/<op_name>/`.

    Returns `None` if the function has no inferable file type (e.g.
    `-> ndarray`). Callers may attach a custom filename_fn out-of-band,
    or skip filename infrastructure entirely.
    """
    file_type = _infer_file_type(func.__wrapped__ if hasattr(func, '__wrapped__') else func)
    if file_type is None:
        return None
    suffix = _FILE_TYPE_SUFFIX.get(file_type.__name__)
    if suffix is None:
        return None
    op_name = func.__name__

    def call_hash_fn(*args, **kwargs):
        return file_type(
            f"{canonical_call_hash(*args, **kwargs)}{suffix}"
        )
    return fully_resolve(call_hash_fn, op_name)


# ---------------------------------------------------------------------
# Image lifecycle (stamp + presentations) — runs as a post-call wrapper
# ---------------------------------------------------------------------

def makes_image_wrap(
    func: Callable,
    *,
    canonical_call_hash: Callable,
) -> Callable:
    """Wrap `func` so that after a successful call:

      1. The result file's `content_hash` header is stamped with
         `canonical_call_hash(*args)` — the canonical call_hash for
         this invocation.
      2. Presentations (tile-pyramid TIFF sibling) are generated.

    Single concern: image lifecycle after the body runs. Doesn't touch
    caching, doesn't attach any attributes to the wrapper, doesn't
    register in `_producers`. Those are owned by `@node` (orchestration)
    and `register_producer` (registry) respectively.
    """
    @wraps(func)
    def wrapped(*args, **kwargs):
        result = func(*args, **kwargs)
        if result is None:
            return result
        call_hash = canonical_call_hash(*args, **kwargs)
        _stamp_content_hash(result, call_hash)
        _generate_presentations(func, result, args, kwargs)
        return result

    return wrapped


def _stamp_content_hash(result: Any, call_hash: str) -> None:
    """Stamp `call_hash` into the result file's native header as
    `content_hash`. Skips non-file results silently. Swallows write
    errors — the system still works if the header write fails (the next
    `get_content_hash` falls back to `hash_of_image_data`)."""
    target = result
    if isinstance(target, WithSidecar):
        target = target.path
    if not isinstance(target, Path):
        return
    try:
        from file_io.generic import set_content_hash
        set_content_hash(target, call_hash)
    except (OSError, ValueError):
        pass


# ---------------------------------------------------------------------
# Presentation generation (tile pyramid)
# ---------------------------------------------------------------------

def _generate_presentations(
    func: Callable, result, args: tuple, kwargs: dict,
) -> None:
    """Auto-generate the sibling pyramidal tile-pyramid TIFF next to
    every file-backed output. That pyramid is the human-facing artifact:
    the web server serves both tiles (1:1 zoom) and thumbnails (a coarse
    pyramid level) from it.

    Lives next to the result file — consumers find it by suffix swap,
    so no separate index is needed.
    """
    if result is None:
        return
    _make_tile_pyramid(result)


def _make_tile_pyramid(result) -> "Path | None":
    """Build a pyramidal tile-pyramid `.tiff` sibling for `result`.

    The pyramid is the single human-facing artifact: a true multi-
    resolution TIFF that backs both 1:1 zoom (tiles) and thumbnails
    (its coarsest level). Always built for file-backed outputs."""
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


def _make_thumbnail(result, resize_array, autostretch, array_to_png, width=1920):
    """Build a 1920px-wide PNG thumbnail from `result`. Used by legacy
    callers; new code reads thumbnails from the tile pyramid's coarsest
    level instead."""
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


# The chain-hook trace function (`_chain_compute_image_metadata`)
# lived here. It was retired 2026-06-09: `compute_image_metadata` now
# composes synchronously inside the `@node(identity=...)` Ray task via
# `decorators._metadata_wrap`, eliminating the
# `_maybe_clean_predecessors` race where the chain-hook Node was the
# producer's immediate dependent and finished too quickly.
