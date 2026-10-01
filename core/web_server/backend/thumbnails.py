import inspect
import os
import threading
from pathlib import Path

import flask
from flask import Blueprint, abort, request

from context import local_context as context
from decisions.resolve import resolve_calibration
from products.registry import _make_thumbnail, producers_for
from web_server.backend._route import finish_route

thumbnails_bp = Blueprint("thumbnails", __name__)


# Per-cache-key single-flight: a thumbnail gallery / tile burst fires
# many simultaneous misses for the same key; without this they'd all
# regenerate the same PNG from disk at once.
_regen_meta_lock = threading.Lock()
_regen_locks: dict = {}


def _key_lock(key) -> threading.Lock:
    with _regen_meta_lock:
        lk = _regen_locks.get(key)
        if lk is None:
            lk = _regen_locks[key] = threading.Lock()
        return lk


def _cache_key(args: tuple) -> str:
    """Stable "_"-joined string key for a set of producer args. Uses
    `stable_hash` when present (the Stack/Mosaic/Target/Run convention)
    and `str(x)` otherwise."""
    parts = []
    for a in args:
        parts.append(getattr(a, "stable_hash", None) or str(a))
    return "_".join(parts)


# Versions the RENDER RECIPE behind every cached thumbnail. Bump when the
# pixel math changes so stale renders regenerate lazily — v2: calibrate_image
# applies partial calibration (2026-09-04; the all-or-nothing guard had every
# flat-corrected Run preview serving the raw light, cached byte-identical
# for every candidate flat).
_THUMBNAIL_VERSION = 2


def _presentation_cache_key(entity, args: tuple, op_name: str) -> tuple:
    """Cache key for a presentation (thumbnail) of an entity produced
    by op `op_name`. Embeds `type(entity).__name__` so distinct kinds
    don't alias."""
    return ("Thumbnail", _THUMBNAIL_VERSION, type(entity).__name__, op_name,
            _cache_key(args))


def _regenerate_thumbnail_for_producer(producer, args: tuple):
    """Rebuild a thumbnail by invoking `producer` for `args`.

    Three producer shapes:
    - File-backed (most ops): `producer.filename_fn(*args)`
      points at an on-disk file we read + downsample.
    - Selection-backed (subframe thumbnails): the filename_fn returns
      an absolute path to a raw input frame.
    - Compute-backed (Run calibration previews): no
      filename_fn — invoke the producer directly to compute an
      ndarray on the fly. Cheap enough; the _is_task guard prevents
      a heavy @node from being re-run as a side effect.

    Returns the PNG bytes, or None if no source resolves."""
    import traceback
    from previews.operations import resize_array, autostretch, array_to_png

    filename_fn = getattr(producer, "filename_fn", None)
    if filename_fn is not None:
        try:
            path = os.fspath(filename_fn(*args))
        except Exception:
            return None
        if not os.path.exists(path):
            return None
        try:
            return _make_thumbnail(
                path, resize_array, autostretch, array_to_png,
            )
        except Exception:
            print(f"### thumbnail regen FAILED ({producer.__name__}): "
                  f"{traceback.format_exc()}")
            return None

    if getattr(producer, "_is_task", False):
        return None
    try:
        result = producer(*args)
        return _make_thumbnail(
            result, resize_array, autostretch, array_to_png,
        )
    except Exception:
        print(f"### thumbnail regen FAILED ({producer.__name__}): "
              f"{traceback.format_exc()}")
        return None


def _thumbnail_for(entity, args: tuple, producer) -> bytes | None:
    """Cached thumbnail bytes for one (entity, args, op), regenerating
    from the on-disk source on a miss (single-flight). None if no
    thumbnail could be built."""
    key = _presentation_cache_key(entity, args, producer.__name__)
    thumbnail = context.get(key)
    if thumbnail:
        return thumbnail
    with _key_lock(key):
        thumbnail = context.get(key)
        if thumbnail:
            return thumbnail
        thumbnail = _regenerate_thumbnail_for_producer(producer, args)
        if thumbnail is not None:
            context[key] = thumbnail
        return thumbnail


def _png_response(thumbnail: bytes, label: str):
    resp = flask.Response(
        thumbnail, mimetype="image/png", direct_passthrough=True,
    )
    resp.headers["X-Producer"] = label
    return resp


def _producer_accepts(producer, nargs: int) -> bool:
    """Whether `producer` can be invoked with exactly `nargs` positional args.

    Producers registered for one entity type can have different arities — a
    Stack has both `stack_files(stack, config)` (2 args) and
    `calibrated_run_preview(stack, run, flat, dark, bias)` (5 args) — and only
    the one whose signature matches the URL's arg count is the intended one.
    Without this filter `get_thumbnail` also invokes the mismatched producer,
    which raises `TypeError` (caught, but it prints a noisy traceback). A
    variadic (`*args`) or un-introspectable producer accepts any count."""
    try:
        params = list(inspect.signature(producer).parameters.values())
    except (TypeError, ValueError):
        return True
    required = optional = 0
    for p in params:
        if p.kind == p.VAR_POSITIONAL:
            return True
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            if p.default is p.empty:
                required += 1
            else:
                optional += 1
    return required <= nargs <= required + optional


def _is_pyramidal(path) -> bool:
    """Whether `path` carries its own multi-resolution pyramid.

    Since "the pyramid IS the payload" a deliverable's primary can be
    pyramidal with no `_tiles.tiff` sibling, so page count — not the presence
    of a sibling file — is what says whether a cheap stored level exists.
    Cheap: a header read, no pixels decoded."""
    try:
        import pyvips
        return int(pyvips.Image.new_from_file(
            str(path), access="sequential").get("n-pages")) > 1
    except Exception:
        return False


def _catalog_thumbnail(entity, args: tuple, extra_args: tuple):
    """Thumbnail via the CATALOG — the object's default FINISH, baked from
    tap snapshots + version-log last-good — the app's own answer to "what
    shows this object right now".

    This is the primary source. Producer `filename_fn`s became call-hash
    based, so they need the op's FULL arguments — `stack_files(stack,
    config, *cal_masters)` — and a bare `/thumbnail/stack/<id>` URL cannot
    supply them: every file-backed producer silently skipped and Stack
    thumbnails 404'd. The catalog needs only (type, stable_hash),
    resolves through the same snapshot/version-log machinery as the viewer,
    and never re-derives a filename without its args — the same "resolve
    versions via the app" rule that keeps stale generations from being
    mistaken for current output.

    Returns PNG bytes or None. Requests with extra producer args (a
    substack) return None — the catalog has no Run products; the producer
    path below still owns those."""
    if extra_args:
        return None
    sid = getattr(entity, "stable_hash", None)
    if not sid:
        return None

    # No presentation-cache layer here: its key carried no input mtime, so a
    # thumbnail cached once was served forever after a rebuild (2026-10-01).
    # `render_finish_preview` already caches durably on the inputs' mtimes;
    # the lock just keeps concurrent first renders single-flight.
    with _key_lock(_presentation_cache_key(entity, args, "catalog")):
        from catalog import default_finish_for, url_type_of
        from finishes.preview import render_finish_preview
        b = default_finish_for(url_type_of(entity), sid)
        if b is None:
            return None
        result = render_finish_preview(b.object_type, b.object_id, b.pipeline, b.name)
        return result[0] if result else None


def get_thumbnail(entity, *extra_args):
    """Best-available thumbnail for an entity + optional extra producer
    args (e.g. filter for Mosaic channels, substack for Runs).

    The CATALOG answers first (`_catalog_thumbnail`): the same tap-snapshot +
    version-log resolution the viewer uses, needing only the entity identity.

    Producer ranking is the fallback: among all producers whose first param
    accepts type(entity) and whose `filename_fn` resolves to an on-disk file,
    pick the one with the LARGEST file (mtime breaks ties — fresher artifact
    wins if sizes match). Bigger file ≈ more pixels ≈ higher quality, so a
    post-processed master shows up above the raw-frame subframe
    thumbnail without any explicit priority list.

    If no producer has written anything, fall back to the compute-on-
    miss path: walk the non-file-backed producers (typically the cheap
    ndarray-returning ones like Run calibration previews) and
    invoke the first that succeeds."""
    args = (entity,) + extra_args

    thumbnail = _catalog_thumbnail(entity, args, extra_args)
    if thumbnail:
        return _png_response(thumbnail, "catalog")
    candidates: list[tuple[int, float, object, str]] = []

    file_backed = []
    compute_backed = []
    for producer in producers_for(type(entity)):
        # Only the producer whose arity matches this URL's arg count is the
        # intended one (e.g. 2-arg uncorrected vs 3-arg flat-corrected Run
        # thumbnails). Skip the rest so we never invoke a producer with the
        # wrong number of args (which raises a noisy, caught TypeError).
        if not _producer_accepts(producer, len(args)):
            continue
        if getattr(producer, "filename_fn", None) is not None:
            file_backed.append(producer)
        else:
            compute_backed.append(producer)

    for producer in file_backed:
        try:
            path = os.fspath(producer.filename_fn(*args))
        except Exception as e:
            # Expected for producers whose arity matches but whose narrow
            # needs more context (e.g. stack_files(stack) without config);
            # logged because a SILENT skip here cost a long debug session.
            print(f"### thumbnail candidate skipped ({producer.__name__}): "
                  f"filename_fn raised {type(e).__name__}: {e}")
            continue
        if not os.path.exists(path):
            print(f"### thumbnail candidate skipped ({producer.__name__}): "
                  f"not on disk: {path}")
            continue
        try:
            size = os.path.getsize(path)
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        candidates.append((size, mtime, producer, path))

    if candidates:
        # Largest, fresher wins.
        candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
        _size, _mtime, producer, path = candidates[0]
        # Prefer a pyramid — serve a small stored level instead of decoding
        # the (possibly multi-GB) source. It lives in one of two places:
        # the `_tiles.tiff` SIBLING, or, since "the pyramid IS the payload"
        # (2026-08-17, `products/registry._publish_with_presentations`),
        # INSIDE the primary itself.
        #
        # Only the sibling was checked. That sibling is no longer written for
        # pyramidal primaries, so every such deliverable fell through to the
        # decode path below — which `_exceeds_thumbnail_budget` refuses at
        # 50 MP, i.e. every real master. The result was a silent 404 where a
        # cheap stored level was sitting in the file all along.
        from previews.create_tile_pyramid import (
            pyramid_path_for, pyramid_thumbnail,
        )
        sibling = pyramid_path_for(Path(path))
        for candidate in (sibling if sibling.exists() else None,
                          Path(path) if _is_pyramidal(path) else None):
            if candidate is None:
                continue
            try:
                return _png_response(
                    pyramid_thumbnail(candidate), producer.__name__)
            except Exception:
                continue
        # No pyramid (e.g. a raw-frame placeholder for an un-built
        # object) — fall back to the lightweight read+resize path.
        thumbnail = _thumbnail_for(entity, args, producer)
        if thumbnail:
            return _png_response(thumbnail, producer.__name__)

    for producer in compute_backed:
        thumbnail = _thumbnail_for(entity, args, producer)
        if thumbnail:
            return _png_response(thumbnail, producer.__name__)

    thumbnail = _stack_subframe_thumbnail(entity, args)
    if thumbnail:
        return _png_response(thumbnail, "subframe")

    return abort(404)


def _stack_subframe_thumbnail(entity, args: tuple):
    """Last resort for a Stack nothing has been built for yet: its best raw
    subframe, downsampled — the original lazy-thumbnail design.

    That design worked through the level-based producer lookup; the call-hash
    registry rewrite silently dropped it, because `get_stack_subframe_image`
    is a plain `@node` (returns a Path, no `identity=`) and so was never an
    artifact producer. Calling its raw body here is a pure metadata lookup —
    frames sorted by star count, no Ray, no pixels — and `_make_thumbnail`
    does the (budget-gated) read."""
    if type(entity).__name__ not in ("Stack", "CalibrationStack"):
        return None

    key = _presentation_cache_key(entity, args, "subframe")
    thumbnail = context.get(key)
    if thumbnail:
        return thumbnail
    with _key_lock(key):
        thumbnail = context.get(key)
        if thumbnail:
            return thumbnail
        try:
            from stacks.compute_graph import get_stack_subframe_image
            path = get_stack_subframe_image._function(entity)
        except Exception:
            return None
        if path is None or not os.path.exists(path):
            return None
        from previews.operations import resize_array, autostretch, array_to_png
        try:
            thumbnail = _make_thumbnail(
                str(path), resize_array, autostretch, array_to_png, width=1200)
        except Exception:
            return None
        if thumbnail is not None:
            context[key] = thumbnail
        return thumbnail


def _finish_png_response(result):
    """PNG + `X-Product-Status` for a rendered finish, or 404. `?fresh=1`:
    strict mode — never serve a last-good stale generation (review
    workflows; the default keeps graceful N/N-1 serving). The status ALWAYS
    travels in a header so image consumers can badge it — previously it was
    computed and silently dropped (2026-08-31)."""
    if result is None:
        return flask.abort(404)
    png, status = result
    if flask.request.args.get("fresh") and status != "live":
        return flask.abort(404)
    resp = flask.Response(png, mimetype="image/png", direct_passthrough=True)
    resp.headers["X-Product-Status"] = status
    return resp


@thumbnails_bp.route("/thumbnail/preview/<object_type>/<object_id>")
def canonical_preview(object_type: str, object_id: str):
    """The object's default finish (its landing view), rendered server-side
    by the SAME traced finish the viewer runs live. 404 when nothing is
    servable yet (the card's onError falls back to the classic thumbnail)."""
    from catalog import default_finish_for
    from finishes.preview import render_finish_preview
    b = default_finish_for(object_type, object_id)
    if b is None:
        return flask.abort(404)
    return _finish_png_response(
        render_finish_preview(b.object_type, b.object_id, b.pipeline, b.name))


@thumbnails_bp.route("/thumbnail/stack/<stack_id>")
def stack_thumbnail(stack_id: str):
    """The stack's card image, and WHICH image it is, in headers:
    `X-Producer: master` -- the default finish rendered from the master
    (`X-Product-Status` live / out_of_date, cached on the input mtimes);
    `X-Producer: subframe` -- nothing stacked yet, so the raw subframe with
    the most stars (`X-Product-Status: absent`); else 404."""
    stack = context.stacks[stack_id].get()
    from catalog import default_finish_for
    from finishes.preview import render_finish_preview
    b = default_finish_for("stack", stack_id)
    result = (
        render_finish_preview(b.object_type, b.object_id, b.pipeline, b.name)
        if b is not None else None
    )
    if result is not None:
        resp = _finish_png_response(result)
        resp.headers["X-Producer"] = "master"
        return resp
    thumbnail = _stack_subframe_thumbnail(stack, (stack,))
    if thumbnail:
        resp = _png_response(thumbnail, "subframe")
        resp.headers["X-Product-Status"] = "absent"
        return resp
    return abort(404)


@thumbnails_bp.route("/thumbnail/target/<target_id>")
def target_thumbnail(target_id: str):
    target = context.targets[target_id].get()
    return get_thumbnail(target)


@finish_route(thumbnails_bp, "/tap/thumbnail", hierarchy=("object_type", "object_id"))
def thumbnail_finish(object_type: str, object_id: str, pipeline: str, finish: str):
    """Thumbnail of one finish of an object under one pipeline — the finish
    baked over a small stored pyramid level of its inputs, never a decode
    of a full-resolution master."""
    from finishes.preview import render_finish_preview
    return _finish_png_response(
        render_finish_preview(object_type, object_id, pipeline, finish))


@thumbnails_bp.route("/thumbnail/calibration_stack/<stack_id>")
def calibration_stack_thumbnail(stack_id: str):
    stack = context.calibration_stacks[stack_id]
    return get_thumbnail(stack)


@thumbnails_bp.route("/thumbnail/run/<stack_id>/<run_id>")
def run_thumbnail(stack_id: str, run_id: str):
    """Preview of one run's best subframe, calibrated as the query says.

        /thumbnail/run/<stack>/<run>?flat=<id>&dark=<id>&bias=<id>

    Every calibration input is NAMED, and what is not named is not applied,
    so the URL alone says what the picture is. `flat=none` pins "no flat"
    explicitly, which is a different statement from omitting it.

    Omitted `dark`/`bias` fall back to what `resolve_calibration` picks for
    this run, NOT to nothing. The positional form this replaced read the
    UNRESOLVED stack, whose run dependencies are always empty, so every
    flat-correction preview divided a frame that still carried its ~87%
    bias+dark pedestal by the flat -- stamping the flat's own pattern onto
    that pedestal at several times the signal amplitude and washing the
    picture out (2026-09-08). Naming the inputs makes that class of bug
    impossible to reintroduce silently.
    """
    stack = context.stacks[stack_id].get()
    resolved = resolve_calibration(
        stack, context.calibration_stacks, context.settings.decision_overrides,
    )
    run = resolved.contents[run_id]

    calibration_stacks = context.calibration_stacks

    def pick(kind: str) -> str | None:
        """Query value wins; else the resolved dependency; `none` means none."""
        arg = request.args.get(kind)
        if arg == "none":
            return None
        if arg:
            if arg not in calibration_stacks:
                abort(404, f"{kind}={arg} is not a calibration stack")
            return arg
        return getattr(run.dependencies.get(kind), "stack_id", None)

    return get_thumbnail(
        resolved, run, pick("flat"), pick("dark"), pick("bias"),
    )


