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
from flask import Blueprint, request, send_file, abort, jsonify

from web_server.backend.router import stateful_route
from web_server.backend._route import tap_route


tiles_bp = Blueprint("tiles", __name__)


# Path -> (large_image handle, source mtime). Cleared when the source
# file's mtime advances. No locking needed because dict assignment is
# atomic in CPython and the worst case is briefly opening a handle twice.
_handle_cache: dict = {}
_handle_cache_lock = threading.Lock()


def _tile_pyramid_path_for_tap(
    object_type: str, object_id: str, pipeline: str, display: str,
    filter: str | None,
):
    """Find the tile pyramid for a specific (pipeline, tap) — the
    `.tiff` sibling of the master file. Path resolution uses
    `_tap_path_unchecked` and gates solely on `.tiff` existence, so a
    tile request still resolves when the master FITS/XISF has been
    reclaimed by ephemeral cleanup."""
    from pathlib import Path
    from web_server.backend.metadata import _tap_path_unchecked
    master = _tap_path_unchecked(
        pipeline, display, object_type, object_id, filter,
    )
    if master is None:
        return None
    tiff = Path(str(master)).with_suffix(".tiff")
    if tiff.exists():
        return str(tiff)
    # Some tap outputs are themselves tiffs (e.g., Subframe mosaics).
    if str(master).endswith((".tiff", ".tif")) and os.path.exists(master):
        return str(master)
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
    tap_route(tiles_bp, "/tap/tile_metadata", hierarchy=("object_type", "object_id")),
    sharding_key="object_id",
)
def tile_metadata_tap(
    object_type: str, object_id: str, pipeline: str, display: str, filter: str,
    backend_port: int = None,
):
    path = _tile_pyramid_path_for_tap(object_type, object_id, pipeline, display, filter)
    return _tile_metadata_response(path, backend_port)


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
    tap_route(tiles_bp, "/tap/tile", hierarchy=("object_type", "object_id")),
    sharding_key="object_id",
)
def serve_tile_tap(
    object_type: str, object_id: str, pipeline: str, display: str, filter: str,
):
    """Single tile from a tap-point's tile pyramid. Tile coords arrive
    as `?z=&x=&y=` so the URL itself stays canonical (URL ends with the
    view triple — see grammar lint)."""
    z = request.args.get("z")
    x = request.args.get("x")
    y = request.args.get("y")
    if z is None or x is None or y is None:
        return abort(400)
    path = _tile_pyramid_path_for_tap(object_type, object_id, pipeline, display, filter)
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
