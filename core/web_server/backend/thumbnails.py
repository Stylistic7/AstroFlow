import inspect
import os
import threading
from pathlib import Path

import flask
from flask import Blueprint, abort

from context import local_context as context
from products.registry import _make_thumbnail, producers_for
from web_server.backend._route import tap_route

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
    - Compute-backed (flat-correction Run thumbnails): no
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
    Run has both `uncorrected_run_thumbnail(stack, substack)` (2 args) and
    `flat_corrected_stack(stack, substack, flat_stack)` (3 args) — and only the
    one whose signature matches the URL's arg count is the intended producer.
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


def _ancestor_chains(entity) -> list[tuple]:
    """Candidate cart-item ancestor chains for a child leaf, root→leaf —
    the same `(type, id)` tuples `parse_ancestors` produces from a URL.

    A Stack/Target has no tap snapshot of its own; only cart items (mosaics,
    standalone targets) do, and `_routes_to_item` filters the ancestor's
    snapshot down to the leaf. The frontend derives ancestors from the URL
    hierarchy; a bare `/thumbnail/<type>/<id>` has no URL to read, but the
    containment is in context. Chains are tried in order until one resolves,
    longest (most specific) first; `()` last so a top-level entity still
    resolves from its own snapshot."""
    sid = getattr(entity, "stable_hash", None)
    kind = type(entity).__name__
    chains: list[tuple] = []
    if kind == "Stack" and sid:
        for tid, tref in sorted((context.targets or {}).items()):
            target = tref.get()
            if sid not in target.stack_ids:
                continue
            for mid, mref in sorted((context.mosaics or {}).items()):
                if tid in mref.get().target_ids:
                    chains.append((("mosaic", mid), ("target", tid)))
            chains.append((("target", tid),))
    elif kind == "Target" and sid:
        for mid, mref in sorted((context.mosaics or {}).items()):
            if sid in mref.get().target_ids:
                chains.append((("mosaic", mid),))
    chains.append(())
    return chains


def _catalog_thumbnail(entity, args: tuple, extra_args: tuple):
    """Thumbnail via the CATALOG — tap snapshots + version-log last-good —
    the app's own answer to "what file shows this object right now".

    This is the primary source. Producer `filename_fn`s became call-hash
    based, so they need the op's FULL arguments — `stack_files(stack,
    config, *cal_masters)` — and a bare `/thumbnail/stack/<id>` URL cannot
    supply them: every file-backed producer silently skipped and Stack
    thumbnails 404'd. The catalog needs only (type, stable_hash, filter),
    resolves through the same snapshot/version-log machinery as the viewer,
    and never re-derives a filename without its args — the same "resolve
    versions via the app" rule that keeps stale generations from being
    mistaken for current output.

    Returns PNG bytes or None. Run-shaped requests (a non-str extra arg,
    e.g. a substack) return None — the catalog has no Run products; the
    producer path below still owns those."""
    if any(not isinstance(a, str) for a in extra_args):
        return None
    sid = getattr(entity, "stable_hash", None)
    if not sid:
        return None

    key = _presentation_cache_key(entity, args, "catalog")
    thumbnail = context.get(key)
    if thumbnail:
        return thumbnail
    with _key_lock(key):
        thumbnail = context.get(key)
        if thumbnail:
            return thumbnail

        from catalog.queries import default_product_for, viewable_path
        path = None
        for ancestors in _ancestor_chains(entity):
            product = default_product_for(
                type(entity).__name__, sid,
                filter=extra_args[0] if extra_args else None,
                ancestors=ancestors)
            if product is None:
                continue
            candidate = viewable_path(product, ancestors)
            if candidate is not None and os.path.exists(candidate):
                path = candidate
                break
        if path is None:
            return None

        png = None
        from previews.create_tile_pyramid import (
            pyramid_path_for, pyramid_thumbnail,
        )
        sibling = pyramid_path_for(Path(path))
        for candidate in (sibling if sibling.exists() else None,
                          Path(path) if _is_pyramidal(path) else None):
            if candidate is None:
                continue
            try:
                png = pyramid_thumbnail(candidate)
                break
            except Exception:
                continue
        if png is None:
            from previews.operations import (
                resize_array, autostretch, array_to_png,
            )
            try:
                png = _make_thumbnail(
                    str(path), resize_array, autostretch, array_to_png)
            except Exception:
                return None
        if png is not None:
            context[key] = png
        return png


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
    ndarray-returning ones like flat-correction Run thumbnails) and
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
                str(path), resize_array, autostretch, array_to_png)
        except Exception:
            return None
        if thumbnail is not None:
            context[key] = thumbnail
        return thumbnail


@thumbnails_bp.route("/thumbnail/preview/<object_type>/<object_id>")
def canonical_preview(object_type: str, object_id: str):
    """The canonical colour preview: autostretched Hubble (+RGB stars when
    present), RGB fallback for broadband-only images — rendered server-side
    by the SAME traced finish the viewer runs live, over a low pyramid level
    of the linear scene formats. 404 when no scene format is servable yet
    (the card's onError falls back to the classic thumbnail)."""
    import flask
    from finishes.preview import render_canonical_preview
    # `?pipeline=` pins which pipeline's scene formats to render. The dashboard
    # passes it so an A/B shows one preview PER pipeline; omitting it keeps the
    # auto-pick used by the object cards.
    result = render_canonical_preview(
        object_type, object_id, pipeline=flask.request.args.get("pipeline"))
    if result is None:
        return flask.abort(404)
    png, status = result
    # `?fresh=1`: strict mode — never serve a last-good stale generation
    # (review workflows; the default keeps graceful N/N-1 serving). The
    # status ALWAYS travels in a header so image consumers can badge it —
    # previously it was computed and silently dropped (2026-08-31).
    if flask.request.args.get("fresh") and status != "live":
        return flask.abort(404)
    resp = flask.Response(png, mimetype="image/png",
                          direct_passthrough=True)
    resp.headers["X-Product-Status"] = status
    return resp


@thumbnails_bp.route("/thumbnail/stack/<stack_id>")
def stack_thumbnail(stack_id: str):
    stack = context.stacks[stack_id].get()
    return get_thumbnail(stack)


@thumbnails_bp.route("/thumbnail/target/<target_id>")
def target_thumbnail(target_id: str):
    target = context.targets[target_id].get()
    return get_thumbnail(target)


@thumbnails_bp.route("/thumbnail/mosaic/<mosaic_id>/<filter>")
def mosaic_thumbnail(mosaic_id: str, filter: str):
    mosaic = context.mosaics[mosaic_id].get()
    return get_thumbnail(mosaic, filter)


@tap_route(thumbnails_bp, "/tap/thumbnail", hierarchy=("object_type", "object_id"))
def thumbnail_tap(
    object_type: str, object_id: str, pipeline: str, display: str, filter: str,
):
    """Thumbnail for a specific (pipeline, tap) of an object.

    Served straight from the tap's pyramidal `.tiff` (a small stored
    level), never by decoding the full-resolution master."""
    from web_server.backend.tiles import _tile_pyramid_path_for_tap
    from previews.create_tile_pyramid import pyramid_thumbnail, NoPreviewAvailable
    tiff = _tile_pyramid_path_for_tap(
        object_type, object_id, pipeline, display, filter,
    )
    if tiff is None:
        return abort(404)
    try:
        thumbnail = pyramid_thumbnail(tiff)
    except NoPreviewAvailable:
        return abort(404)  # no pyramid yet → no preview yet (never full-load)
    except Exception:
        return abort(500)
    return _png_response(thumbnail, display)


@thumbnails_bp.route("/thumbnail/calibration_stack/<stack_id>")
def calibration_stack_thumbnail(stack_id: str):
    stack = context.calibration_stacks[stack_id]
    return get_thumbnail(stack)


@thumbnails_bp.route("/thumbnail/run/<stack_id>/<substack_id>")
def uncorrected_run_thumbnail_route(stack_id: str, substack_id: str):
    stack = context.stacks[stack_id].get()
    substack = stack.contents[substack_id]
    return get_thumbnail(stack, substack)


@thumbnails_bp.route("/thumbnail/run/<stack_id>/<substack_id>/<flat_stack_id>")
def run_thumbnail(stack_id: str, substack_id: str, flat_stack_id: str):
    stack = context.stacks[stack_id].get()
    substack = stack.contents[substack_id]
    flat_stack = context.calibration_stacks[flat_stack_id]
    return get_thumbnail(stack, substack, flat_stack)


