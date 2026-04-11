"""HTTP endpoints for serving image tiles to the web frontend.

Tile pyramids are produced ahead of time as a side effect of the image
producers themselves (via @create_tile_pyramid). This module just looks
them up via `producer.tile_pyramid_fn(*args)` and serves them through
large_image. There is no on-demand conversion: if a tile pyramid
doesn't exist for a given object/level, the request returns 404.

A small in-memory cache holds opened large_image handles, keyed by the
on-disk tile pyramid path. The handle is invalidated when the source
file's mtime advances (which happens when a producer rewrites its
output in place — the @makes_image sidecar-discard pattern handles the
cache-invalidation chain upstream of this layer).
"""
import io
import os
import threading

import large_image
from flask import Blueprint, Response, send_file, abort, jsonify, request
from flask_cors import cross_origin

from context import context
from previews.types import QualityLevel
from stacks.composite_stack import Stack, CalibrationStack
from targets.types import Target
from mosaics.types import Mosaic
from products.registry import get_producer, MediaType
from web_server.backend.router import stateful_route


tiles_bp = Blueprint("tiles", __name__)


# Path -> (large_image handle, source mtime). Cleared when the source
# file's mtime advances. No locking needed because dict assignment is
# atomic in CPython and the worst case is briefly opening a handle twice.
_handle_cache: dict = {}
_handle_cache_lock = threading.Lock()


_TYPE_MAP = {
    "stack": Stack,
    "calibration_stack": CalibrationStack,
    "target": Target,
    "mosaic": Mosaic,
}


def _resolve_obj_args(object_type: str, object_id: str, filter: str | None):
    """Look up the object and return the (obj_type, obj_args) tuple
    that the producer's tile_pyramid_fn expects as its first arg(s)."""
    obj_cls = _TYPE_MAP.get(object_type)
    if obj_cls is None:
        raise ValueError(f"Unknown object_type {object_type}")

    if object_type == "stack":
        return obj_cls, (context.stacks[object_id].get(),)
    if object_type == "calibration_stack":
        return obj_cls, (context.calibration_stacks[object_id],)
    if object_type == "target":
        return obj_cls, (context.targets[object_id].get(), filter)
    if object_type == "mosaic":
        return obj_cls, (context.mosaics[object_id].get(), filter)


def _tile_pyramid_path_for(
    object_type: str, object_id: str, filter: str | None, level: str | None,
):
    """Find the tile pyramid for an object at a specific level (or the
    best available level when level is None or 'auto'). Returns the
    on-disk path or None if no producer has yet generated a tile pyramid
    for this object/level."""
    obj_cls, obj_args = _resolve_obj_args(object_type, object_id, filter)

    if level and level != "auto":
        levels_to_try = [QualityLevel(level)]
    else:
        # Best-quality first, falling back to lower quality.
        levels_to_try = [
            QualityLevel.POST_PROCESSED,
            QualityLevel.MASTER,
            QualityLevel.DOWNSCALED_MASTER,
            QualityLevel.SUBFRAME,
        ]

    for q in levels_to_try:
        producer = get_producer(MediaType.IMAGE, obj_cls, q)
        if producer is None:
            continue
        if not hasattr(producer, "tile_pyramid_fn"):
            continue
        path = producer.tile_pyramid_fn(*obj_args)
        if path is not None and os.path.exists(path):
            return path

    return None


def _get_handle(tile_pyramid_path):
    """Open a large_image handle for a tile pyramid path, with mtime-based
    invalidation. Re-opens if the file has been rewritten since we last
    cached it."""
    path_str = os.fspath(tile_pyramid_path)
    current_mtime = os.path.getmtime(path_str)

    with _handle_cache_lock:
        cached = _handle_cache.get(path_str)
        if cached is not None:
            handle, cached_mtime = cached
            if cached_mtime == current_mtime:
                return handle

        handle = large_image.open(path_str, plugin="vips")
        _handle_cache[path_str] = (handle, current_mtime)
        return handle


@stateful_route(
    tiles_bp.route("/tile_metadata/<object_type>/<object_id>/<filter>/<level>"),
    sharding_key="object_id",
)
def tile_metadata(
    object_type: str, object_id: str, filter: str, level: str = "auto",
    backend_port: int = None,
):
    path = _tile_pyramid_path_for(object_type, object_id, filter, level)
    if path is None:
        return abort(404)

    handle = _get_handle(path)
    meta = handle.getMetadata()
    return jsonify({
        "width": meta["sizeX"],
        "height": meta["sizeY"],
        "tileWidth": meta["tileWidth"],
        "tileHeight": meta["tileHeight"],
        "maxZoom": meta["levels"] - 1,
        "port": backend_port,
    })


@tiles_bp.route(
    "/tile/<object_type>/<object_id>/<filter>/<level>/<int:z>/<int:x>/<int:y>.png"
)
@cross_origin(origins="*")
def serve_tile(
    object_type: str, object_id: str, filter: str, level: str,
    z: str, x: str, y: str,
):
    path = _tile_pyramid_path_for(object_type, object_id, filter, level)
    if path is None:
        return abort(404)

    try:
        handle = _get_handle(path)
        tile_data = handle.getTile(x, y, z, encoding="PNG")
        return send_file(io.BytesIO(tile_data), mimetype="image/png")
    except Exception as e:
        print(f"Error serving tile for {object_type, object_id}: {e}")
        return abort(500)
