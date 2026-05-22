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
from flask import Blueprint, send_file, abort, jsonify

from context import local_context as context
from previews.types import QualityLevel, QUALITY_PRIORITY, PIXINSIGHT
from stacks.composite_stack import Stack, CalibrationStack
from targets.types import Target
from mosaics.types import Mosaic
from products.registry import get_producer, MediaType
from products.identity import (
    StackId, CalibrationStackId, TargetChannelId, MosaicChannelId,
)
from web_server.backend.router import stateful_route


tiles_bp = Blueprint("tiles", __name__)


# Path -> (large_image handle, source mtime). Cleared when the source
# file's mtime advances. No locking needed because dict assignment is
# atomic in CPython and the worst case is briefly opening a handle twice.
_handle_cache: dict = {}
_handle_cache_lock = threading.Lock()


_TYPE_MAP = {
    "stack": StackId,
    "calibration_stack": CalibrationStackId,
    "target": TargetChannelId,
    "mosaic": MosaicChannelId,
}


def _resolve_obj_args(object_type: str, object_id: str, filter: str | None):
    """Look up the object and return the (identity_cls, obj_args) tuple
    that the producer's filename_fn / tile_pyramid_fn expects."""
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
    implementation: str | None = None,
):
    """Find the tile pyramid for an object at a specific level (or the
    best available level when level is None or 'auto'). Returns the
    on-disk path or None if no producer has yet generated a tile pyramid
    for this object/level. If implementation is None, uses the active
    setting."""
    obj_cls, obj_args = _resolve_obj_args(object_type, object_id, filter)

    if level and level != "auto":
        levels_to_try = [level]
    else:
        levels_to_try = QUALITY_PRIORITY

    # `implementation` URL param is optional; default to PixInsight,
    # the only backend in production today.
    impl = implementation or PIXINSIGHT

    from pathlib import Path
    from products.registry import get_filename_fn

    for q in levels_to_try:
        producer = get_producer(MediaType.IMAGE, obj_cls, q, impl)
        if producer is None:
            continue
        # Legacy path: @create_tile_pyramid attaches `tile_pyramid_fn`
        # directly to the producer.
        if hasattr(producer, "tile_pyramid_fn"):
            path = producer.tile_pyramid_fn(*obj_args, level=q)
            if path is not None and os.path.exists(path):
                return path
        # Newer path: @makes_image with presentations=[TILE_PYRAMID]
        # writes a .tiff sibling next to the master's main output but
        # doesn't attach a lookup function — find the file by name.
        master_fn = get_filename_fn(MediaType.IMAGE, obj_cls, q, impl)
        if master_fn is not None:
            try:
                master_path = master_fn(*obj_args)
            except (TypeError, KeyError):
                master_path = None
            if master_path is not None:
                tiff = Path(str(master_path)).with_suffix(".tiff")
                if tiff.exists():
                    return tiff

    # Fallback: in-progress merge tile pyramids. merge_images writes
    # these at a deterministic path per (mosaic, filter, mode),
    # overwriting on each merge step — surfaces partial mosaic results
    # before the final store_mosaic_color_channel completes.
    if obj_cls is MosaicChannelId:
        output_dir = context.output_dir
        if output_dir is not None:
            mosaic_obj, mosaic_filter = obj_args[0], obj_args[1]
            for q in levels_to_try:
                progress = Path(output_dir) / "Mosaic" / str(q) / f"progress_{mosaic_obj.stable_hash}_{mosaic_filter}.tiff"
                if progress.exists():
                    return str(progress)

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
    return _tile_metadata_response(
        _tile_pyramid_path_for(object_type, object_id, filter, level),
        backend_port,
    )


@stateful_route(
    tiles_bp.route("/tile_metadata/<object_type>/<object_id>/<implementation>/<level>/<filter>"),
    sharding_key="object_id",
)
def tile_metadata_specific(
    object_type: str, object_id: str, implementation: str, level: str, filter: str,
    backend_port: int = None,
):
    """tile_metadata pinned to a specific implementation. Legacy URL
    shape; new callers should use the tap-keyed route below."""
    return _tile_metadata_response(
        _tile_pyramid_path_for(object_type, object_id, filter, level, implementation),
        backend_port,
    )


@stateful_route(
    tiles_bp.route("/tap/tile_metadata/<object_type>/<object_id>/<path:pipeline_and_display>/<filter>"),
    sharding_key="object_id",
)
def tile_metadata_tap(
    object_type: str, object_id: str, pipeline_and_display: str, filter: str,
    backend_port: int = None,
):
    """tile_metadata for a specific (pipeline, tap) of an object.

    `pipeline_and_display` is `<pipeline>/<display>` — joined into one
    path segment so Flask doesn't have to know that pipeline names
    contain spaces. Tile-pyramid paths are looked up via the
    PipelineOutputIndex (recorded at trace time) with a static-mapping
    fallback.
    """
    from web_server.backend.metadata import tap_path_for
    pipeline, _, display = pipeline_and_display.rpartition("/")
    if not pipeline or not display:
        return abort(404)
    path = _tap_tile_pyramid_path(pipeline, display, object_type, object_id, filter)
    return _tile_metadata_response(path, backend_port)


def _tap_tile_pyramid_path(pipeline, display, object_type, object_id, filter):
    """Tile-pyramid TIFF for a tap point — same sibling-of-master
    pattern as the legacy lookup, but the master comes from
    tap_path_for() instead of (impl, level) filename_fn."""
    from pathlib import Path
    from web_server.backend.metadata import tap_path_for
    master = tap_path_for(pipeline, display, object_type, object_id, filter)
    if master is None:
        return None
    tiff = Path(str(master)).with_suffix(".tiff")
    if tiff.exists():
        return str(tiff)
    return master if master.endswith((".tiff", ".tif")) else None


def _tile_metadata_response(path, backend_port):
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
        "mtime": os.path.getmtime(path),
    })


@stateful_route(
    tiles_bp.route(
        "/tile/<object_type>/<object_id>/<filter>/<level>/<int:z>/<int:x>/<int:y>.png"
    ),
    sharding_key="object_id",
)
def serve_tile(
    object_type: str, object_id: str, filter: str, level: str,
    z: str, x: str, y: str,
):
    path = _tile_pyramid_path_for(object_type, object_id, filter, level)
    return _serve_tile_response(path, object_type, object_id, x, y, z)


@stateful_route(
    tiles_bp.route(
        "/tile/<object_type>/<object_id>/<implementation>/<level>/<filter>/<int:z>/<int:x>/<int:y>.png"
    ),
    sharding_key="object_id",
)
def serve_tile_specific(
    object_type: str, object_id: str, implementation: str, level: str, filter: str,
    z: str, x: str, y: str,
):
    """Tile pinned to a specific implementation. Legacy URL shape;
    new callers should use the tap-keyed route below."""
    path = _tile_pyramid_path_for(object_type, object_id, filter, level, implementation)
    return _serve_tile_response(path, object_type, object_id, x, y, z)


@stateful_route(
    tiles_bp.route(
        "/tap/tile/<object_type>/<object_id>/<path:pipeline_and_display>/<filter>/<int:z>/<int:x>/<int:y>.png"
    ),
    sharding_key="object_id",
)
def serve_tile_tap(
    object_type: str, object_id: str, pipeline_and_display: str, filter: str,
    z: str, x: str, y: str,
):
    """Single tile from a tap-point's tile pyramid."""
    pipeline, _, display = pipeline_and_display.rpartition("/")
    if not pipeline or not display:
        return abort(404)
    path = _tap_tile_pyramid_path(pipeline, display, object_type, object_id, filter)
    return _serve_tile_response(path, object_type, object_id, x, y, z)


def _serve_tile_response(path, object_type, object_id, x, y, z):
    if path is None:
        return abort(404)

    try:
        handle = _get_handle(path)
        tile_data = handle.getTile(x, y, z, encoding="PNG")
        return send_file(io.BytesIO(tile_data), mimetype="image/png")
    except Exception as e:
        print(f"Error serving tile for {object_type, object_id}: {e}")
        return abort(500)
