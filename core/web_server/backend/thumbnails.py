import io
import os
import dataclasses
import threading
from pathlib import Path

import flask
from flask import Blueprint, abort

from context import local_context as context
from products.registry import _make_thumbnail, producers_for, MediaType
from products.identity import (
    ObjectId, PresentationType, StackId, CalibrationStackId, TargetId,
    MosaicChannelId,
    CalibratedStackId, UncorrectedRunId,
)

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


def _regenerate_thumbnail_for_producer(producer, identity: ObjectId):
    """Rebuild a thumbnail by invoking `producer` for `identity`.

    Three producer shapes:
    - File-backed (most ops): `producer.filename_fn(*identity_args)`
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

    args = [getattr(identity, f.name) for f in dataclasses.fields(identity)]

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
            print(f"### thumbnail regen FAILED ({producer.__name__}, "
                  f"{type(identity).__name__}): {traceback.format_exc()}")
            return None

    if getattr(producer, "_is_task", False):
        return None
    try:
        result = producer(*args)
        return _make_thumbnail(
            result, resize_array, autostretch, array_to_png,
        )
    except Exception:
        print(f"### thumbnail regen FAILED ({producer.__name__}, "
              f"{type(identity).__name__}): {traceback.format_exc()}")
        return None


def _thumbnail_for(identity: ObjectId, producer) -> bytes | None:
    """Cached thumbnail bytes for one (identity, op), regenerating
    from the on-disk source on a miss (single-flight). None if no
    thumbnail could be built."""
    key = identity.presentation_cache_key(
        PresentationType.THUMBNAIL, producer.__name__,
    )
    thumbnail = context.get(key)
    if thumbnail:
        return thumbnail
    with _key_lock(key):
        thumbnail = context.get(key)
        if thumbnail:
            return thumbnail
        thumbnail = _regenerate_thumbnail_for_producer(producer, identity)
        if thumbnail is not None:
            context[key] = thumbnail
        return thumbnail


def _png_response(thumbnail: bytes, label: str):
    resp = flask.Response(
        thumbnail, mimetype="image/png", direct_passthrough=True,
    )
    resp.headers["X-Producer"] = label
    return resp


def get_thumbnail_by_id(identity: ObjectId):
    """Best-available thumbnail for `identity`.

    Ranking: among all `@makes_image`-registered producers whose
    `filename_fn` resolves to an on-disk file, pick the one with
    the LARGEST file (mtime breaks ties — fresher artifact wins if
    sizes match). Bigger file ≈ more pixels ≈ higher quality, so
    a post-processed master shows up above the raw-frame subframe
    thumbnail without any explicit priority list.

    If no producer has written anything, fall back to the
    compute-on-miss path: walk the non-file-backed producers
    (typically the cheap ndarray-returning ones like
    flat-correction Run thumbnails) and invoke the first that
    succeeds."""
    candidates: list[tuple[int, float, ObjectId, object]] = []
    args = [getattr(identity, f.name) for f in dataclasses.fields(identity)]

    file_backed = []
    compute_backed = []
    for producer in producers_for(type(identity)):
        if getattr(producer, "filename_fn", None) is not None:
            file_backed.append(producer)
        else:
            compute_backed.append(producer)

    for producer in file_backed:
        try:
            path = os.fspath(producer.filename_fn(*args))
        except Exception:
            continue
        if not os.path.exists(path):
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
        # Prefer the pyramidal `.tiff` sibling — serve a small stored
        # level instead of decoding the (possibly multi-GB) source.
        tiff = Path(path).with_suffix(".tiff")
        if tiff.exists():
            from previews.create_tile_pyramid import pyramid_thumbnail
            try:
                return _png_response(pyramid_thumbnail(tiff), producer.__name__)
            except Exception:
                pass
        # No pyramid (e.g. a raw-frame placeholder for an un-built
        # object) — fall back to the lightweight read+resize path.
        thumbnail = _thumbnail_for(identity, producer)
        if thumbnail:
            return _png_response(thumbnail, producer.__name__)

    for producer in compute_backed:
        thumbnail = _thumbnail_for(identity, producer)
        if thumbnail:
            return _png_response(thumbnail, producer.__name__)

    return abort(404)


@thumbnails_bp.route("/thumbnail/stack/<stack_id>")
def stack_thumbnail(stack_id: str):
    stack = context.stacks[stack_id].get()
    return get_thumbnail_by_id(StackId(stack=stack))


@thumbnails_bp.route("/thumbnail/target/<target_id>")
def target_thumbnail(target_id: str):
    target = context.targets[target_id].get()
    return get_thumbnail_by_id(TargetId(target=target))


@thumbnails_bp.route("/thumbnail/mosaic/<mosaic_id>/<filter>")
def mosaic_thumbnail(mosaic_id: str, filter: str):
    mosaic = context.mosaics[mosaic_id].get()
    return get_thumbnail_by_id(MosaicChannelId(mosaic=mosaic, filter=filter))


@thumbnails_bp.route("/tap/thumbnail/<object_type>/<object_id>/<path:pipeline_and_display>/<filter>")
@thumbnails_bp.route("/tap/thumbnail/<object_type>/<object_id>/<path:pipeline_and_display>")
def thumbnail_tap(
    object_type: str, object_id: str, pipeline_and_display: str,
    filter: str | None = None,
):
    """Thumbnail for a specific (pipeline, tap) of an object.

    Served straight from the tap's pyramidal `.tiff` (a small stored
    level), never by decoding the full-resolution master.
    `<path:pipeline_and_display>` is `<pipeline>/<display>` joined
    into one URL segment so Flask doesn't need to know that pipeline
    display names contain spaces."""
    from web_server.backend.tiles import _tile_pyramid_path_for_tap
    from previews.create_tile_pyramid import pyramid_thumbnail
    pipeline, _, display = pipeline_and_display.rpartition("/")
    if not pipeline or not display:
        return abort(404)
    tiff = _tile_pyramid_path_for_tap(
        object_type, object_id, pipeline, display, filter,
    )
    if tiff is None:
        return abort(404)
    try:
        thumbnail = pyramid_thumbnail(tiff)
    except Exception:
        return abort(500)
    return _png_response(thumbnail, display)


@thumbnails_bp.route("/thumbnail/calibration_stack/<stack_id>")
def calibration_stack_thumbnail(stack_id: str):
    stack = context.calibration_stacks[stack_id]
    return get_thumbnail_by_id(CalibrationStackId(stack=stack))


@thumbnails_bp.route("/thumbnail/run/<stack_id>/<substack_id>")
def uncorrected_run_thumbnail_route(stack_id: str, substack_id: str):
    stack = context.stacks[stack_id].get()
    substack = stack.contents[substack_id]
    return get_thumbnail_by_id(
        UncorrectedRunId(stack=stack, substack=substack),
    )


@thumbnails_bp.route("/thumbnail/run/<stack_id>/<substack_id>/<flat_stack_id>")
def run_thumbnail(stack_id: str, substack_id: str, flat_stack_id: str):
    stack = context.stacks[stack_id].get()
    substack = stack.contents[substack_id]
    flat_stack = context.calibration_stacks[flat_stack_id]
    return get_thumbnail_by_id(
        CalibratedStackId(stack=stack, substack=substack, flat_stack=flat_stack),
    )


def get_video(identity: type, *args):
    """First producer registered for VIDEO + `identity` that yields
    a video stream. Each video producer attaches a `video_fn`
    callable that, given the identity args, returns bytes or None."""
    for producer in producers_for(identity, MediaType.VIDEO):
        video_fn = getattr(producer, "video_fn", None)
        if video_fn is None:
            continue
        video = video_fn(*args)
        if video is None:
            continue
        buffer = io.BytesIO()
        buffer.write(video)
        buffer.seek(0)
        return flask.send_file(buffer, mimetype="video/webm")
    return abort(404)
