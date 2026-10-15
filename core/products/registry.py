"""Image-lifecycle and producer-registry helpers.

Public surface — all consumed by `@node` (decorators/decorators.py),
which is the sole orchestrator that calls these in the right order
with the right shared artifacts.

Helpers:

  * `build_filename_fn(func, canonical_call_hash)` — construct the
    resolved `filename_fn` for an artifact-producing op. Returns one
    function (no polymorphic dispatcher — every @node has exactly one
    routing path).
  * `makes_image_wrap(func, *, filename_fn, canonical_call_hash)` — own
    an image producer's output: hand the body a scratch `out`, then stamp
    the header, build the tile pyramid, record the returned sidecar
    sections and land the file with one atomic rename.
  * `register_producer(wrapper, func)` — add the final wrapper to
    `_producers[type_of_first_param]` for view-side enumeration.

Also kept here: the producer registry itself (`_producers`,
`producers_for`), file-type inference utilities, and the tile pyramid
generator.
"""
import inspect
import os
import tempfile
import uuid
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np
import structlog

from file_io.types import ContentHashable, Produced, WithSidecar
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
    from file_io.types import output_file_type
    file_type = output_file_type(func)
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
    filename_fn: Callable,
    canonical_call_hash: Callable,
    makes_tiff: bool = True,
    image_type: str | None = "light",
) -> Callable:
    """Own an image producer's output from scratch file to published product.

    The body is called with `out=`: a scratch path beside the product's final,
    content-addressed path (same directory, so same filesystem, same suffix).
    It writes the image there and returns `Produced(out, header=, sections=)`
    — or None when it produces nothing. It never learns the final name, never
    touches a sidecar and never publishes. Then, in this order:

      1. the header is stamped in one write: CALLHASH, plus IMAGETYP/STBLHASH
         when the body set none, the cards an output inherits from its
         inputs, and the body's own `header` cards (`_stamp_header`);
      2. a `display=` op's tile pyramid is built — as the payload itself for a
         TIFF, as a `_tiles.tiff` sibling otherwise;
      3. the final path's sidecar takes the body's `sections`, plus the canvas
         the image is drawn on (`mosaics.canvas_stamp`) — merged under the
         sidecar lock, and before the payload lands, so a published file's
         sidecar is never missing what its producer recorded;
      4. the payload lands with ONE `os.replace` — the final path is never
         touched before, so a reader sees the previous product or the new
         one, never a partial, and a killed run leaves only scratch files.

    The sections that describe the old bytes (`image_metadata`, `broken`) are
    dropped in step 3: the metadata wrap measures the new file at once."""
    params = [p for p in inspect.signature(func).parameters.values() if p.name != "out"]

    @wraps(func)
    def wrapped(*args, **kwargs):
        from file_io.sidecar import sidecar_path_for
        final = filename_fn(*args, **kwargs)
        final.parent.mkdir(parents=True, exist_ok=True)
        out = final.with_name(f".{final.stem}.partial.{os.getpid()}.{uuid.uuid4().hex[:8]}{final.suffix}")
        litter = (out, out.with_name(out.name + ".pyr"), sidecar_path_for(out))
        try:
            result = func(*args, out=out, **kwargs)
            if result is None:
                return None
            assert isinstance(result, Produced) and Path(result.path) == out, (
                f"{func.__name__} must return Produced(out, ...) for the `out` it was "
                f"given, not {result!r}")
            assert out.exists(), f"{func.__name__} returned Produced(out) but wrote nothing at {out}"
            assert not sidecar_path_for(out).exists(), (
                f"{func.__name__} wrote a sidecar beside its scratch output; return "
                "its sections in Produced(sections=...) instead")
            _publish(out, final, result, canonical_call_hash(*args, **kwargs),
                     image_type, args, kwargs, makes_tiff)
        finally:
            for f in litter:
                f.unlink(missing_ok=True)
        return final

    wrapped.__signature__ = inspect.signature(func).replace(parameters=params)
    return wrapped


def _publish(out: Path, final: Path, result, call_hash: str, image_type,
             args: tuple, kwargs: dict, makes_tiff: bool) -> None:
    """Steps 1-4 of `makes_image_wrap`. A failed stamp or pyramid build still
    publishes the planar payload: the master is expensive, its presentation is
    not (`_make_tile_pyramid` logs and swallows its own failures)."""
    from file_io.sidecar import update_sidecar
    from mosaics.canvas_stamp import SECTION as CANVAS, canvas_for_call
    from previews.create_tile_pyramid import pyramid_path_for
    payload = out
    try:
        _stamp_header(out, call_hash, image_type, args, result.header)
        if makes_tiff and _pyramid_is_the_payload(out):
            # THE PYRAMID IS THE PAYLOAD (2026-08-17): a pyramidal TIFF's page 0
            # is bit-identical to the planar master, so storing both wasted
            # ~265 GB at full build. Land the pyramid as the product.
            pyr = out.with_name(out.name + ".pyr")
            _make_tile_pyramid(out, dest=pyr, served=final)
            if pyr.exists() and pyr.stat().st_size > 0:
                payload = pyr
        elif makes_tiff:
            _make_tile_pyramid(out, dest=pyramid_path_for(final))
    finally:
        sections = dict(result.sections)
        canvas = canvas_for_call(args, kwargs)
        if canvas is not None:
            sections[CANVAS] = canvas
            sibling = pyramid_path_for(final)       # serves when the primary is reclaimed
            if sibling.exists():
                update_sidecar(sibling, lambda root: root["astroflow"].update({CANVAS: canvas}))

        def _record(root):
            for stale in ("image_metadata", "broken"):
                root["astroflow"].pop(stale, None)
            root["astroflow"].update(sections)
        update_sidecar(final, _record)
        os.replace(payload, final)


def _primary_stable_hash(args: tuple) -> "str | None":
    """The producer's primary identity: the first call arg carrying a
    `stable_hash`. Same convention as trace_paths/preflight entity
    extraction — the identity axis of every product file."""
    for a in args:
        sh = getattr(a, "stable_hash", None)
        if sh is not None:
            return str(sh)
    return None


# Cards an output of the SAME field and filter shares with its input. RA,
# DEC, FOCALLEN and XPIXSZ are the plate solver's SEARCH PRIORS (see
# plate_solving._header_hints): they turn a blind all-sky quad match into a
# 5-degree cone search. FILTER is the image's band: a deconvolved or
# starless channel is still that channel (2026-10-14: every BXT and SXT
# output read FILTER='none').
_INHERITED_KEYS = ("FILTER", "RA", "DEC", "FOCALLEN", "XPIXSZ")


def _missing(header, key: str) -> bool:
    # A header read through `normalize_header` reports an absent FILTER as
    # "none" — absent all the same.
    value = header.get(key)
    return value in (None, "") or (key == "FILTER" and str(value).lower() == "none")


def _inherited_cards(header, args) -> dict:
    """Cards copied from the producing op's INPUT image, for outputs
    written by external tools that drop them.

    rc-astro's BXT writes a TIFF with no RA/DEC/FOCALLEN/XPIXSZ, so
    `bxt_channel` outputs faced the solver completely blind — and on a
    field the blind ladder cannot crack (vdB15, 2026-08-22) that is a hard
    failure after two 600 s CPU rungs, repeated every drain. The panel is at
    the same place on the sky as its input, so the input's cards are exactly
    the right priors. This does NOT reuse the input's SOLUTION: BXT's
    aberration correction shifts the geometry, so the re-solve still happens
    and still produces the corrected WCS (Croman) — it just knows where to
    look.

    Only fills cards the output is MISSING, so a producer that writes its
    own always wins."""
    hints: dict = {}
    missing = [k for k in _INHERITED_KEYS if _missing(header, k)]
    if not missing:
        return hints
    from metadata.classes import ImageMetadata
    images = [getattr(a, "header", None) or {} for a in args if isinstance(a, ImageMetadata)]
    for key in missing:
        if key == "FILTER":
            # A band is inherited only from inputs that AGREE on one: an
            # H+O narrowband composite is neither (it read FILTER='H' from
            # its first input, 2026-10-14).
            bands = {str(h["FILTER"]) for h in images if not _missing(h, "FILTER")}
            if len(bands) == 1:
                hints[key] = bands.pop()
            continue
        for src in images:
            if not _missing(src, key):
                hints[key] = src.get(key)
                break
    return hints


def _stamp_header(
    result: Any, call_hash: str, image_type: str | None,
    args: tuple = (), cards: "dict | None" = None,
) -> None:
    """Stamp the result file's native header in a single write:

      * `CALLHASH` = `call_hash` — always. Provenance axis +
        write-completion witness (stamped only after the body finished, so a
        body that writes straight to its final path cannot be mistaken for
        finished when its process was killed). Preflight compares this.
        CONTHASH is NOT written here: it is optional, it means "my producer
        declared an identity of its own", and only a body that wants one
        writes it — `_stamped_or_hashed` falls back to CALLHASH otherwise,
        which is the value every consumer saw before the 2026-09-22 split.
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
        from file_io.generic import read_header, update_header, _CALL_HASH_KEY
        header = read_header(target)
        updates: dict[str, Any] = {}
        if image_type is not None and not header.get("IMAGETYP"):
            updates["IMAGETYP"] = image_type
        stable = _primary_stable_hash(args)
        if stable is not None and not header.get("STBLHASH"):
            updates["STBLHASH"] = stable
        updates |= _inherited_cards(header, args)
        updates |= dict(cards or {})            # the body's own cards win over defaults
        updates[_CALL_HASH_KEY] = str(call_hash)
        update_header(target, updates)
    except (OSError, ValueError):
        pass


# ---------------------------------------------------------------------
# Presentation generation (tile pyramid)
# ---------------------------------------------------------------------

def _pyramid_is_the_payload(source: Path) -> bool:
    """Whether this product should BE its pyramid rather than have one.

    True only for TIFF display taps. Two exclusions, both deliberate:

      * NON-DISPLAY TIFFs (crop_panel, bxt_channel, sxt_channel, nxt, ...) get
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


def _make_tile_pyramid(result, dest: "Path | None" = None,
                       served: "Path | None" = None) -> "Path | None":
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
    # `dest` lets the caller build from the producer's scratch file while
    # landing the pyramid at the FINAL product's sibling path — see
    # `makes_image_wrap`. `default_make_tile_pyramid` writes
    # `<dest>.tmp` then `os.replace`s, so the pyramid itself lands atomically.
    tile_path = Path(dest) if dest is not None else pyramid_path_for(source)
    try:
        default_make_tile_pyramid(source, tile_path, served=served)
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
