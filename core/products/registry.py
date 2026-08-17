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
import structlog

from file_io.types import ContentHashable, WithSidecar
from context import context


log = structlog.get_logger()


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
        # Selection-style fns (`filename_fn=get_best_light_frame`) return the
        # op body's own value — WithSidecar(path). The filename_fn contract
        # is the PATH: unwrap, or the os.fspath below raises and every
        # caller that resolves this op's output (the lazy-thumbnail
        # endpoint's file-backed probe, resolve_output_paths) silently
        # treats the op as pathless.
        from file_io.types import WithSidecar as _WS
        if isinstance(filename, _WS):
            filename = filename.path
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
    makes_tiff: bool = True,
    image_type: str | None = "light",
) -> Callable:
    """Wrap `func` so that after a successful call:

      1. The result file's native header is stamped: `content_hash` =
         `canonical_call_hash(*args)` always, plus `IMAGETYP` = `image_type`
         when the op declares one and the body didn't already set IMAGETYP.
      2. If `makes_tiff` (the op has `display=`), the tile-pyramid TIFF
         sibling is generated — the web-openable co-representation of the
         primary. Non-display intermediates skip it (no tap, not shown).

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
        if makes_tiff:
            _publish_with_presentations(result, call_hash, image_type, args)
        else:
            _stamp_header(result, call_hash, image_type, args)
        return result

    return wrapped


def _primary_stable_hash(args: tuple) -> "str | None":
    """The producer's primary identity: the first call arg carrying a
    `stable_hash`. Same convention as trace_paths/preflight entity
    extraction — the identity axis of every product file."""
    for a in args:
        sh = getattr(a, "stable_hash", None)
        if sh is not None:
            return str(sh)
    return None


def _stamp_header(
    result: Any, call_hash: str, image_type: str | None,
    args: tuple = (),
) -> None:
    """Stamp the result file's native header in a single write:

      * `content_hash` (CONTHASH) = `call_hash` — always. Provenance axis +
        write-completion witness (stamped only after the body finished).
      * `STBLHASH` = the primary arg's stable_hash — when resolvable AND the
        body didn't already write one (the stackers/mergers stamp their own,
        sometimes composed differently — never clobber). Identity axis:
        without it, identity-addressed consumers (master matching, catalog,
        version log, migrations) go trace-dependent for the file.
      * `IMAGETYP` = `image_type` — only when the op declares one AND the
        body didn't already set IMAGETYP. The "don't clobber" rule lets the
        shared stacker keep its calibration types ("Master Dark" etc.) while
        every other producer defaults to a light frame, so `is_light_frame`
        is uniformly correct for produced images.

    One write, not three: `update_xisf_header` rewrites the whole file, so
    folding the keys into a single `update_header` avoids redundant
    full-file rewrites. Skips non-file results silently; swallows write
    errors (the next `get_content_hash` falls back to `hash_of_image_data`)."""
    target = result
    if isinstance(target, WithSidecar):
        target = target.path
    if not isinstance(target, Path):
        return
    try:
        from file_io.generic import read_header, update_header, _CONTENT_HASH_KEY
        header = read_header(target)
        updates: dict[str, Any] = {_CONTENT_HASH_KEY: call_hash}
        if image_type is not None and not header.get("IMAGETYP"):
            updates["IMAGETYP"] = image_type
        stable = _primary_stable_hash(args)
        if stable is not None and not header.get("STBLHASH"):
            updates["STBLHASH"] = stable
        update_header(target, updates)
    except (OSError, ValueError):
        pass


# ---------------------------------------------------------------------
# Presentation generation (tile pyramid)
# ---------------------------------------------------------------------

def _result_path(result) -> "Path | None":
    """The on-disk path a producer's result refers to, or None if it isn't
    file-backed. Same unwrapping `_stamp_header` and `_make_tile_pyramid` do."""
    if isinstance(result, np.ndarray):
        return None
    if isinstance(result, WithSidecar):
        result = result.path
    if isinstance(result, tempfile._TemporaryFileWrapper):
        return Path(result.name)
    try:
        return Path(os.fspath(result))
    except TypeError:
        return None


def _publish_with_presentations(
    result, call_hash: str, image_type, args: tuple,
) -> None:
    """Stamp and build the display pyramid while the payload is UNPUBLISHED,
    then land the pyramid FIRST and the payload LAST.

    Why: the body publishes the payload at its final content-addressed path,
    but the pyramid is a separate ~15-minute pass afterwards. For that whole
    window the product was RESOLVABLE BUT UNTILEABLE — and on a rebuild the
    new payload had already destroyed the working previous one. On 2026-08-14
    a γ Cyg `Linear RGB` master sat in exactly that state and served 36,150
    broken tiles.

    Renaming the payload aside first shrinks that window from ~15 minutes to
    the microseconds between the body's own `os.replace` and this one, and
    establishes the invariant **payload present => pyramid present**. During a
    rebuild the PREVIOUS `_tiles.tiff` stays in place and keeps serving (the
    tile route resolves through the pyramid), so viewers see the old image
    until the new pair is complete instead of a black page.

    The staging name PRESERVES THE SUFFIX: header reads, `update_header` and
    `needs_display_pyramid` all dispatch on extension, so `.tiff` must stay
    `.tiff` while parked.
    """
    source = _result_path(result)
    if source is None or not source.exists():
        # Not file-backed, or the body wrote somewhere other than the path we
        # can see. Don't silently skip the pyramid — fall back to the old
        # in-place order so behaviour is unchanged for these producers.
        _stamp_header(result, call_hash, image_type, args)
        _generate_presentations(None, result, args, {})
        return

    from previews.create_tile_pyramid import pyramid_path_for
    staging = source.with_name(
        f".{source.stem}.staging.{os.getpid()}{source.suffix}")
    os.replace(source, staging)          # unpublish — atomic, instant
    published = staging                  # what will finally land at `source`
    scratch = None
    try:
        _stamp_header(staging, call_hash, image_type, args)
        # `_make_tile_pyramid` swallows+logs build failures by design (the
        # pyramid is optional; failing must not discard the expensive master).
        if _pyramid_is_the_payload(staging):
            # THE PYRAMID IS THE PAYLOAD (2026-08-17). A pyramidal TIFF's page 0
            # is bit-identical to the planar master — verified maxdiff 0.0 at
            # five crops, image-description preserved — so storing both wastes
            # ~265 GB at full build. Build it beside the staging file and land
            # THAT as the product; no `_tiles.tiff` sibling is written.
            #
            # The I/O is identical to what already happened: this same
            # read-back-and-write-pyramid ran every publish, just to a sibling
            # path. Only the destination and the discarded planar copy differ.
            scratch = staging.with_name(staging.name + ".pyr")
            _make_tile_pyramid(staging, dest=scratch)
            if scratch.exists() and scratch.stat().st_size > 0:
                published = scratch
        else:
            _make_tile_pyramid(staging, dest=pyramid_path_for(source))
    finally:
        # ALWAYS republish, even if stamping or the pyramid raised. Otherwise
        # the payload is stranded under its dot-name and the product vanishes
        # entirely — strictly worse than a missing pyramid. On the pyramidal
        # path a failed build leaves `published is staging`, so the planar
        # master lands and the product is still correct, just unpyramided.
        os.replace(published, source)    # payload lands LAST
        if published is not staging:
            staging.unlink(missing_ok=True)
        elif scratch is not None:
            scratch.unlink(missing_ok=True)


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


def _pyramid_is_the_payload(source: Path) -> bool:
    """Whether this product should BE its pyramid rather than have one.

    True only for TIFF display taps. Two exclusions, both deliberate:

      * NON-DISPLAY TIFFs (crop_panel, bxt_correct, sxt_starless, nxt, ...) get
        no pyramid at all today, and giving them one would ADD ~156 GB of mip
        levels nobody reads. `needs_display_pyramid` already draws that line.
        It also keeps the rc-astro inputs planar: `crop_panel` feeds BXT/SXT/NXT
        as CLI tools whose tiled-TIFF support is unverified, and PixInsight —
        measured 2026-08-17 — silently returns ZERO windows on tiled input.
      * NON-TIFF payloads (FITS) keep a sibling pyramid; a TIFF pyramid cannot
        be the payload for a format the pipeline must hand to other tools.
        (XISF needs no pyramid at all — see `file_io/xisf_mmap`.)
    """
    from previews.create_tile_pyramid import needs_display_pyramid
    return (source.suffix.lower() in (".tif", ".tiff")
            and needs_display_pyramid(source))


def _make_tile_pyramid(result, dest: "Path | None" = None) -> "Path | None":
    """Build the display tile pyramid — a separate `<stem>_tiles.tiff`
    sibling — for `result`.

    The pyramid is the web-serving artifact backing both 1:1 zoom
    (tiles) and thumbnails (its coarsest level). It lives in its own
    `_tiles.tiff` namespace so the deliverable can itself be a `.tiff`
    (a first-class output format) without the pyramid aliasing it.
    Always built for file-backed outputs."""
    if isinstance(result, np.ndarray):
        return None
    if isinstance(result, tempfile._TemporaryFileWrapper):
        source = Path(result.name)
    else:
        source = Path(os.fspath(result))
    if not source.exists():
        return None
    from previews.create_tile_pyramid import (
        default_make_tile_pyramid, pyramid_path_for, needs_display_pyramid,
    )
    # Already-display-ready uint8 tiffs (Subframe previews) are their own
    # display artifact and serve directly — no separate pyramid.
    if not needs_display_pyramid(source):
        return source
    # `dest` lets the caller build from a PARKED payload (a `.staging.` name)
    # while still landing the pyramid at the FINAL product's sibling path —
    # see `_publish_with_presentations`. `default_make_tile_pyramid` writes
    # `<dest>.tmp` then `os.replace`s, so the pyramid itself lands atomically.
    tile_path = Path(dest) if dest is not None else pyramid_path_for(source)
    try:
        default_make_tile_pyramid(source, tile_path)
    except Exception as e:
        # Non-fatal by design: the pyramid is an optional display artifact,
        # and failing here would throw away the (expensive) master the body
        # just produced. But it must NOT be invisible — a swallowed build
        # failure leaves a master with no pyramid, which surfaces as a 404
        # in the viewer with no other trace. Log loud + structured so it's
        # greppable (historically the OOM on multi-GB masters landed here).
        log.warning(
            "tile_pyramid_build_failed",
            source=str(source), tile_path=str(tile_path),
            error=str(e), error_type=type(e).__name__,
        )
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
        from file_io.generic import read_image, read_image_geometry
        from previews.create_tile_pyramid import _exceeds_thumbnail_budget
        # No pyramid sibling exists (that path is preferred upstream) — so this
        # would decode the full deliverable. Refuse if it's over budget: a
        # gigapixel master has no preview until its pyramid is built, rather
        # than ballooning the serving process (see _THUMBNAIL_MAX_SRC_PX).
        try:
            g = read_image_geometry(Path(filename))
            if _exceeds_thumbnail_budget(g.width_pixels, g.height_pixels):
                return None
        except Exception:
            pass
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
