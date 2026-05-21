import io
import os
import dataclasses
import threading

import flask
from flask import Blueprint, abort

from context import local_context as context
from stacks.composite_stack import Stack
from targets.types import Target
from mosaics.types import Mosaic
from mosaics.merging import MergedImage
from products.registry import (
    get_producer, get_filename_fn, fully_resolve, _make_thumbnail, MediaType,
)
from products.identity import (
    ObjectId, PresentationType, StackId, CalibrationStackId, TargetId,
    MosaicChannelId,
    CalibratedStackId, UncorrectedRunId,
)
from previews.types import QualityLevel, PIXINSIGHT, QUALITY_PRIORITY, SUBFRAME,MASTER,POST_PROCESSED
from stacks.run import Run
from stacks.composite_stack import CalibrationStack

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


def _regenerate_thumbnail(
    identity: ObjectId,
    level: str, implementation: str,
):
    """Rebuild a thumbnail from the producer's on-disk source, or None
    if nothing resolvable/extant exists for this (type, level, impl).
    Lazy replacement for eager generation: the thumbnail cache is now
    the bounded evictable store, so an evicted entry self-heals on view.

    Three producer shapes:
    - master-backed (Stack/Mosaic MASTER/POST_PROCESSED): filename_fn
      yields a name resolved under output_dir/Type/Level/Impl.
    - selection-backed (SUBFRAME stack_thumbnail/calibration_stack_
      thumbnail): filename_fn (get_best_light_frame/get_first_frame) is
      itself the cheap selection and returns an absolute raw-frame path.
    - compute-backed (flat-correction Run thumbnails): NO filename_fn;
      the producer returns a computed ndarray and persists nothing.
      Invoke it on miss — safe because it's cheap (one calibrated
      subframe) and directly callable; the _is_task guard guarantees
      we never re-run an expensive @node (a file-backed producer always
      has a filename_fn and never reaches that branch anyway)."""
    import traceback
    from previews.operations import resize_array, autostretch, array_to_png
    identity_cls = type(identity)
    args = [getattr(identity, f.name) for f in dataclasses.fields(identity)]

    filename_fn = get_filename_fn(
        MediaType.IMAGE, identity_cls, level, implementation,
    )
    if filename_fn is not None:
        candidates = []
        try:
            candidates.append(os.fspath(fully_resolve(
                identity_cls, level, filename_fn, implementation,
            )(*args)))
        except Exception:
            pass
        try:
            candidates.append(os.fspath(filename_fn(*args)))
        except Exception:
            pass
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            return None
        try:
            return _make_thumbnail(path, resize_array, autostretch, array_to_png)
        except Exception:
            print(f"### thumbnail regen FAILED ({identity_cls.__name__}, "
                  f"{level}, {implementation}): {traceback.format_exc()}")
            return None

    # No filename_fn → compute-on-miss.
    producer = get_producer(
        MediaType.IMAGE, identity_cls, level, implementation,
    )
    if producer is None or getattr(producer, "_is_task", False):
        return None
    try:
        result = producer(*args)
        return _make_thumbnail(result, resize_array, autostretch, array_to_png)
    except Exception:
        print(f"### thumbnail regen FAILED ({identity_cls.__name__}, "
              f"{level}, {implementation}): {traceback.format_exc()}")
        return None


def _thumbnail_for(
    identity: ObjectId,
    level: str, implementation: str,
):
    """Cached thumbnail bytes for one (level, impl), regenerating from
    the on-disk master on a miss (single-flight). None if unavailable."""
    key = identity.presentation_cache_key(
        PresentationType.THUMBNAIL, level, implementation,
    )
    thumbnail = context.get(key)
    if thumbnail:
        return thumbnail
    with _key_lock(key):
        thumbnail = context.get(key)  # filled while we waited?
        if thumbnail:
            return thumbnail
        thumbnail = _regenerate_thumbnail(
            identity, level, implementation,
        )
        if thumbnail is not None:
            context[key] = thumbnail
        return thumbnail


def _png_response(thumbnail: bytes, level: str):
    resp = flask.Response(
        thumbnail, mimetype="image/png", direct_passthrough=True,
    )
    resp.headers["X-Quality-Level"] = str(level)
    return resp


def get_thumbnail_by_id(identity: ObjectId):
    """Return the best-available thumbnail using ObjectId-based lookup."""
    implementation = PIXINSIGHT
    for level in QUALITY_PRIORITY:
        thumbnail = _thumbnail_for(
            identity, level, implementation,
        )
        if thumbnail:
            return _png_response(thumbnail, level)
    return abort(404)


def get_specific_thumbnail_by_id(
    implementation: str,
    level: str, identity: ObjectId,
):
    """Return the thumbnail for an exact (implementation, level)."""
    thumbnail = _thumbnail_for(identity, level, implementation)
    if thumbnail:
        return _png_response(thumbnail, level)
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


@thumbnails_bp.route("/thumbnail/mosaic/<mosaic_id>/<implementation>/<quality>/<filter>")
def mosaic_thumbnail_specific(mosaic_id: str, implementation: str, quality: str, filter: str):
    """Thumbnail for an exact (mosaic, implementation, quality, filter)."""
    mosaic = context.mosaics[mosaic_id].get()
    return get_specific_thumbnail_by_id(
        implementation, quality,
        MosaicChannelId(mosaic=mosaic, filter=filter),
    )


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
    implementation = PIXINSIGHT
    for level in QUALITY_PRIORITY:
        if fn := get_producer(MediaType.VIDEO, identity, level, implementation):
            if hasattr(fn, "video_fn"):
                if video := fn.video_fn(*args):
                    buffer = io.BytesIO()
                    buffer.write(video)
                    buffer.seek(0)
                    return flask.send_file(buffer, mimetype="video/webm")

    return abort(404)
