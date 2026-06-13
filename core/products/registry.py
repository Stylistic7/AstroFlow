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
    Drops in for the old `identity=[StackId, CalibrationStackId]`
    polymorphism.
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


def image_producer(*, display: str | None = None, final: bool = False,
                   identity=None):
    """Lightweight `@image_producer` for non-cached, non-Ray functions
    that still need to be discoverable via `producers_for(type_of_first_arg)`.

    `identity=` is accepted for legacy-call-site compatibility but
    ignored — routing keys off the first-parameter type now.
    """
    del identity  # vestigial
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

    Previously the path included `<object_type>/` derived from a
    per-op ObjectId class — but that was a redundant decorative
    layer once `Processed/` stopped being a human-browse target.
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


def pick_identity(identities: list, first_arg, op_name: str | None = None):
    """Pick the identity class whose `object_type` matches `first_arg`.

    Falls back to `identities[0]` when no isinstance check matches —
    the phantom-marker case (e.g. `MergedImageId.object_type =
    MergedImage`, which no real first_arg ever instances). Warns when
    the fallback fires AND the op was registered with more than one
    identity, since silent fallback there would misroute to whichever
    identity happened to be first in the registration order.

    Returns None when `identities` is empty.
    """
    if not identities:
        return None
    for ic in identities:
        ot = getattr(ic, "object_type", None)
        if ot is not None and isinstance(first_arg, ot):
            return ic
    if len(identities) > 1:
        import structlog
        structlog.get_logger().warning(
            "products.pick_identity: no isinstance match on multi-identity op; "
            "falling back to identities[0] — taps may misroute. Pick "
            "identity explicitly or split the op.",
            op=op_name, identities=[ic.__name__ for ic in identities],
            first_arg_type=type(first_arg).__name__,
        )
    return identities[0]


def _make_dispatcher(filename_fns: dict) -> Callable:
    """Pick the right filename_fn by finding the first arg whose type
    matches one of the registered identities.

    `stack_files` registers for both `StackId` and `CalibrationStackId`;
    each carries its own resolved fn pointing at a different
    `<object_type>` directory. The call_hash inside is identical (same
    args, same hash), only the routing differs.

    Scans all positional args (not just `args[0]`) so the literals-first
    signature convention works: `stack_files(stack, config, *cal_masters)`
    picks the identity matching `stack` (at args[1]), not `config`.
    """
    identities = list(filename_fns.keys())

    def dispatch(*args, **kwargs):
        # Direct isinstance scan over args — bypass pick_identity's
        # "fallback to identities[0]" warning, because here a mismatch
        # just means "this arg isn't the entity, keep looking." Also
        # check `arg.entity` so trace_paths' `_TracedOutput` proxies
        # (which wrap a dep's content_hash + entity) route by their
        # underlying identity instead of misrouting via the fallback.
        for a in args:
            candidate = a if not hasattr(a, "entity") else (a.entity or a)
            for ic in identities:
                ot = getattr(ic, "object_type", None)
                if ot is not None and isinstance(candidate, ot):
                    return filename_fns[ic](*args, **kwargs)
        # No arg matched — let pick_identity handle the warn-and-
        # fallback path with the first arg, preserving the old
        # diagnostic behaviour for genuinely ambiguous calls.
        if not args:
            for fn in filename_fns.values():
                return fn(*args, **kwargs)
            raise RuntimeError("no filename_fn registered")
        identity_cls = pick_identity(identities, args[0])
        if identity_cls is None:
            raise RuntimeError("no filename_fn registered")
        return filename_fns[identity_cls](*args, **kwargs)
    return dispatch


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
