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
from mosaics.merge_images import MergedImage
from products.registry import (
    get_producer, get_filename_fn, fully_resolve, _make_thumbnail, MediaType,
)
from products.identity import (
    ObjectId, PresentationType, StackId, TargetId,
    MosaicChannelId,
    CalibratedStackId, UncorrectedRunId,
)
from previews.types import QualityLevel, Implementation, QUALITY_PRIORITY
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
    object_type: type, identity: ObjectId,
    level: QualityLevel, implementation: Implementation,
):
    """Rebuild a thumbnail from the on-disk master, or None if there's
    no resolvable/extant master for this (type, level, impl). Lazy
    replacement for eager generation: the thumbnail cache is now the
    bounded evictable store, so an evicted entry self-heals on view."""
    filename_fn = get_filename_fn(
        MediaType.IMAGE, object_type, level, implementation,
    )
    if filename_fn is None:
        return None
    args = [getattr(identity, f.name) for f in dataclasses.fields(identity)]
    try:
        master = fully_resolve(
            object_type, level, filename_fn, implementation,
        )(*args)
    except Exception:
        return None
    path = os.fspath(master)
    if not os.path.exists(path):
        return None
    from previews.operations import resize_array, autostretch, array_to_png
    try:
        return _make_thumbnail(path, resize_array, autostretch, array_to_png)
    except Exception:
        return None


def _thumbnail_for(
    object_type: type, identity: ObjectId,
    level: QualityLevel, implementation: Implementation,
):
    """Cached thumbnail bytes for one (level, impl), regenerating from
    the on-disk master on a miss (single-flight). None if unavailable."""
    key = identity.presentation_cache_key(
        PresentationType.THUMBNAIL, object_type, level, implementation,
    )
    thumbnail = context.get(key)
    if thumbnail:
        return thumbnail
    with _key_lock(key):
        thumbnail = context.get(key)  # filled while we waited?
        if thumbnail:
            return thumbnail
        thumbnail = _regenerate_thumbnail(
            object_type, identity, level, implementation,
        )
        if thumbnail is not None:
            context[key] = thumbnail
        return thumbnail


def _png_response(thumbnail: bytes, level: QualityLevel):
    resp = flask.Response(
        thumbnail, mimetype="image/png", direct_passthrough=True,
    )
    resp.headers["X-Quality-Level"] = level.value
    return resp


def get_thumbnail_by_id(object_type: type, identity: ObjectId):
    """Return the best-available thumbnail using ObjectId-based lookup."""
    implementation = context.settings.implementation
    for level in QUALITY_PRIORITY:
        thumbnail = _thumbnail_for(
            object_type, identity, level, implementation,
        )
        if thumbnail:
            return _png_response(thumbnail, level)
    return abort(404)


def get_specific_thumbnail_by_id(
    object_type: type, implementation: Implementation,
    level: QualityLevel, identity: ObjectId,
):
    """Return the thumbnail for an exact (implementation, level)."""
    thumbnail = _thumbnail_for(object_type, identity, level, implementation)
    if thumbnail:
        return _png_response(thumbnail, level)
    return abort(404)


@thumbnails_bp.route("/thumbnail/stack/<stack_id>")
def stack_thumbnail(stack_id: str):
    stack = context.stacks[stack_id].get()
    return get_thumbnail_by_id(Stack, StackId(stack=stack))


@thumbnails_bp.route("/thumbnail/target/<target_id>")
def target_thumbnail(target_id: str):
    target = context.targets[target_id].get()
    return get_thumbnail_by_id(Target, TargetId(target=target))


@thumbnails_bp.route("/thumbnail/mosaic/<mosaic_id>/<filter>")
def mosaic_thumbnail(mosaic_id: str, filter: str):
    mosaic = context.mosaics[mosaic_id].get()
    return get_thumbnail_by_id(Mosaic, MosaicChannelId(mosaic=mosaic, filter=filter))


@thumbnails_bp.route("/thumbnail/mosaic/<mosaic_id>/<implementation>/<quality>/<filter>")
def mosaic_thumbnail_specific(mosaic_id: str, implementation: str, quality: str, filter: str):
    """Thumbnail for an exact (mosaic, implementation, quality, filter)."""
    mosaic = context.mosaics[mosaic_id].get()
    return get_specific_thumbnail_by_id(
        Mosaic, Implementation(implementation), QualityLevel(quality),
        MosaicChannelId(mosaic=mosaic, filter=filter),
    )


@thumbnails_bp.route("/thumbnail/calibration_stack/<stack_id>")
def calibration_stack_thumbnail(stack_id: str):
    stack = context.calibration_stacks[stack_id]
    return get_thumbnail_by_id(CalibrationStack, StackId(stack=stack))


@thumbnails_bp.route("/thumbnail/run/<stack_id>/<substack_id>")
def uncorrected_run_thumbnail_route(stack_id: str, substack_id: str):
    stack = context.stacks[stack_id].get()
    substack = stack.contents[substack_id]
    return get_thumbnail_by_id(
        Run, UncorrectedRunId(stack=stack, substack=substack),
    )


@thumbnails_bp.route("/thumbnail/run/<stack_id>/<substack_id>/<flat_stack_id>")
def run_thumbnail(stack_id: str, substack_id: str, flat_stack_id: str):
    stack = context.stacks[stack_id].get()
    substack = stack.contents[substack_id]
    flat_stack = context.calibration_stacks[flat_stack_id]
    return get_thumbnail_by_id(
        Run, CalibratedStackId(stack=stack, substack=substack, flat_stack=flat_stack),
    )


def get_video(object_type: type, *args):
    implementation = context.settings.implementation
    for level in QUALITY_PRIORITY:
        if fn := get_producer(MediaType.VIDEO, object_type, level, implementation):
            if hasattr(fn, "video_fn"):
                if video := fn.video_fn(*args):
                    buffer = io.BytesIO()
                    buffer.write(video)
                    buffer.seek(0)
                    return flask.send_file(buffer, mimetype="video/webm")

    return abort(404)
