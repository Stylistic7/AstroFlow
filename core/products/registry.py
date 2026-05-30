"""Image-lifecycle and producer-registry helpers.

Public surface — all consumed by `@node` (decorators/decorators.py),
which is the sole orchestrator that calls these in the right order with
the right shared artifacts.

Each helper has a single concern. None of them queries another's state
via attribute lookup; everything is passed in explicitly. The reason is
historical: the old `@makes_image` decorator coordinated with `@node`
through attribute attachment and closure-timing assumptions, which
silently produced wrong call_hashes (see 2026-05-29).

Helpers:

  * `validate_identities(func, identities)` — raise `TypeError` if any
    identity's fields aren't a subset of the function's parameters.
  * `build_filename_fns(func, identities, canonical_call_hash)` —
    construct one resolved `filename_fn` per identity plus a polymorphic
    dispatcher. Returns `(filename_fns_dict, dispatcher_fn)`.
  * `makes_image_wrap(func, *, canonical_call_hash, identities)` —
    wrap `func` so that after the function returns, the result file's
    `content_hash` header is stamped to the call_hash and presentation
    artifacts (tile pyramid sibling) are generated. Single concern:
    image lifecycle after the body runs.
  * `register_producer(wrapper, identities, media_type=MediaType.IMAGE)` —
    add the final wrapper to the `_producers[media_type][identity]`
    list for view-side enumeration.

Also kept here: the producer registry itself (`_producers`,
`producers_for`), file-type inference utilities, the tile pyramid
generator, and the trace hook that auto-chains `compute_image_metadata`.
"""
import dataclasses
import inspect
import os
import tempfile
import typing
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np

from file_io.types import ContentHashable, WithSidecar
from products.identity import ObjectId, PresentationType
from context import context


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


type FilenameFn = Union[str, Callable[..., Path]]


# ---------------------------------------------------------------------
# Producer registry
# ---------------------------------------------------------------------

# `_producers[media_type][identity]` — every op `@node(identity=X)`
# appends itself so view-side endpoints can enumerate "every op that
# produces an X-shaped image."
_producers: dict[MediaType, dict[type, list[Callable]]] = {
    MediaType.IMAGE: {},
    MediaType.VIDEO: {},
}

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


def producers_for(
    identity: type[ObjectId], media_type: MediaType = MediaType.IMAGE,
) -> list[Callable]:
    """Every op registered to produce `identity`-shaped artifacts of
    `media_type`. Returns the wrappers in registration order."""
    _ensure_producers_imported()
    return list(_producers[media_type].get(identity, []))


def register_producer(
    wrapper: Callable,
    identities: list[type[ObjectId]],
    media_type: MediaType = MediaType.IMAGE,
) -> None:
    """Add `wrapper` to `_producers[media_type][identity]` for each
    identity. The wrapper is the FINAL `@node`-wrapped callable (so
    view-side lookups get `.remote`, `.filename_fn`, etc.).

    Single concern: registry maintenance. No filename construction, no
    validation — those happen elsewhere."""
    for identity in identities:
        _producers[media_type].setdefault(identity, []).append(wrapper)


def image_producer(
    identity: type[ObjectId],
    *,
    display: str | None = None,
    final: bool = False,
):
    """Lightweight `@image_producer(identity=X)` for non-cached, non-Ray
    functions that still need to be discoverable via `producers_for(X)`.

    Use this for ndarray-returning preview helpers called inline from
    `@node` tasks. The wrapper function is identical to the original
    (no extra runtime behavior) — it just attaches `_identities` and
    registers in `_producers` so the view-side enumeration treats it
    as a producer of `X`-shaped images.

    Most ops should use `@node(identity=X, ...)` instead, which builds
    the full cached / file-routing / lifecycle pipeline. Reserve this
    for the rare op that genuinely doesn't need any of that machinery.
    """
    def decorator(func):
        func._identities = (
            getattr(func, '_identities', []) + [identity]
        )
        if display is not None:
            func._display = display
        if final:
            func._final = True
        register_producer(func, [identity])
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


def fully_resolve(
    identity_cls: type, filename_fn: FilenameFn, op_name: str,
):
    """Wrap `filename_fn` so its output is prefixed with the output-dir
    routing path:

        <output_dir>/<object_type>/<op_name>/<filename_fn(...)>

    `op_name` (the producer's `__name__`) is the on-disk directory;
    different ops writing the same identity-shape land in different
    directories. Selection-style absolute paths are passed through
    unchanged (used for raw-frame producers)."""
    object_type_name = identity_cls.object_type.__name__

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
        return filename.resolve_at(output_dir / object_type_name / op_name)

    return inner


def _extract_identity(
    identity_cls: type[ObjectId],
    func: Callable, args: tuple, kwargs: dict,
) -> ObjectId:
    """Pull identity-field values out of a call's bound args."""
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    id_fields = {f.name for f in dataclasses.fields(identity_cls)}
    id_kwargs = {name: bound.arguments[name] for name in id_fields}
    return identity_cls(**id_kwargs)


# ---------------------------------------------------------------------
# Validation + filename construction (called by @node)
# ---------------------------------------------------------------------

def validate_identities(func: Callable, identities: list[type[ObjectId]]) -> None:
    """Raise TypeError if any identity has fields that aren't parameter
    names on `func`. Called by `@node` at decoration time.

    `@makes_image` cannot be applied to a Ray-task wrapper (the order
    needs `@node` outside), so the older `_is_task` check is no longer
    relevant — `@node` is the sole entry point now."""
    raw_func = func.__wrapped__ if hasattr(func, '__wrapped__') else func
    producer_params = set(inspect.signature(raw_func).parameters.keys())
    for identity in identities:
        id_fields = {f.name for f in dataclasses.fields(identity)}
        missing = id_fields - producer_params
        if missing:
            raise TypeError(
                f"{identity.__name__} has fields {missing} that don't "
                f"match any parameter of "
                f"{getattr(func, '__name__', '...')}. ObjectId field "
                f"names must be a subset of producer parameter names."
            )


def build_filename_fns(
    func: Callable,
    identities: list[type[ObjectId]],
    canonical_call_hash: Callable,
) -> tuple[dict, Callable] | tuple[None, None]:
    """Construct one resolved `filename_fn` per identity, plus a
    polymorphic dispatcher.

    Each per-identity filename_fn:
      1. Calls `canonical_call_hash(*args, **kwargs)` to get the
         op's call_hash for THIS call.
      2. Returns `<call_hash>.<ext>`.
      3. `fully_resolve` wraps that to prepend the output-dir routing.

    The dispatcher picks the right per-identity fn by inspecting the
    first arg's type (`isinstance(first, identity.object_type)`).
    Polymorphic ops (e.g. `stack_files` registered for both StackId and
    CalibrationStackId) dispatch by Stack vs CalibrationStack at call time.

    Returns `(None, None)` if the function has no inferable file type
    (e.g. `-> ndarray`). In that case the caller may attach a custom
    filename_fn out-of-band, or skip filename infrastructure entirely.
    """
    file_type = _infer_file_type(func.__wrapped__ if hasattr(func, '__wrapped__') else func)
    if file_type is None:
        return None, None
    suffix = _FILE_TYPE_SUFFIX.get(file_type.__name__)
    if suffix is None:
        return None, None
    op_name = func.__name__

    def call_hash_fn(*args, **kwargs):
        return file_type(
            f"{canonical_call_hash(*args, **kwargs)}{suffix}"
        )

    filename_fns = {
        identity: fully_resolve(identity, call_hash_fn, op_name)
        for identity in identities
    }
    return filename_fns, _make_dispatcher(filename_fns)


def _make_dispatcher(filename_fns: dict) -> Callable:
    """Pick the right filename_fn by inspecting the first arg's type.

    `stack_files` registers for both `StackId` and `CalibrationStackId`;
    each carries its own resolved fn pointing at a different
    `<object_type>` directory. The call_hash inside is identical (same
    args, same hash), only the routing differs."""
    def dispatch(*args, **kwargs):
        if not args:
            for fn in filename_fns.values():
                return fn(*args, **kwargs)
            raise RuntimeError("no filename_fn registered")
        first = args[0]
        for identity_cls, resolved in filename_fns.items():
            ot = identity_cls.object_type
            if ot is None:
                continue
            if isinstance(first, ot):
                return resolved(*args, **kwargs)
        for fn in filename_fns.values():
            return fn(*args, **kwargs)
        raise RuntimeError("no filename_fn matched")
    return dispatch


# ---------------------------------------------------------------------
# Image lifecycle (stamp + presentations) — runs as a post-call wrapper
# ---------------------------------------------------------------------

def makes_image_wrap(
    func: Callable,
    *,
    canonical_call_hash: Callable,
    identities: list[type[ObjectId]],
) -> Callable:
    """Wrap `func` so that after a successful call:

      1. The result file's `content_hash` header is stamped with
         `canonical_call_hash(*args)` — the canonical call_hash for
         this invocation.
      2. The result's stale sidecar is discarded (if any).
      3. Presentations (tile-pyramid TIFF sibling) are generated.

    Single concern: image lifecycle after the body runs. Doesn't touch
    caching, doesn't attach any attributes to the wrapper, doesn't
    register in `_producers`. Those are owned by `@node` (orchestration)
    and `register_producer` (registry) respectively.
    """
    from file_io.sidecar import discard_sidecar

    @wraps(func)
    def wrapped(*args, **kwargs):
        result = func(*args, **kwargs)
        if result is None:
            return result
        # Stamp content_hash = canonical_call_hash(args). Pure Merkle
        # convention: every @node-produced file's header carries the
        # call_hash that produced it, so downstream consumers reading
        # content_hash get a value derivable from the trace alone.
        call_hash = canonical_call_hash(*args, **kwargs)
        _stamp_content_hash(result, call_hash)
        if not isinstance(result, np.ndarray):
            discard_sidecar(result)
        _generate_presentations(func, result, args, kwargs, identities)
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
    identities: list[type],
) -> None:
    """Auto-generate the sibling pyramidal tile-pyramid TIFF next to
    every file-backed output. That pyramid is the human-facing artifact:
    the web server serves both tiles (1:1 zoom) and thumbnails (a coarse
    pyramid level) from it.

    Iterated over each identity the op produces — for polymorphic ops
    the same result is keyed under each identity it can be looked up by."""
    if result is None:
        return
    tile_path = _make_tile_pyramid(result)
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


# ---------------------------------------------------------------------
# Trace-time hooks
# ---------------------------------------------------------------------

def _chain_compute_image_metadata(handle):
    """Trace-time hook: every `@node(identity=...)` op produces a file
    whose canonical downstream consumer is its `ImageMetadata`, so we
    auto-chain a `compute_image_metadata` node after the producer.
    User pipelines skip the `compute_image_metadata(stack(...))`
    boilerplate."""
    from dependency_graph.tracing import record
    from metadata.image_metadata import compute_image_metadata
    return record(compute_image_metadata, handle)
