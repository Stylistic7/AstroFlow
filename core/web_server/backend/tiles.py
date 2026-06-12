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

Three endpoints:

* `/tap/tile_metadata` — image dimensions, tile size, level count.
* `/tap/tile/.../{z}/{x}/{y}.f32` — raw float32 little-endian pixels,
  brotli-compressed (`Content-Encoding: br`). Each tile is read out of
  the pyramid via `large_image.getTile(..., format=TILE_FORMAT_NUMPY)`,
  packed as raw bytes, brotli-compressed on the fly, and shipped.
* `/tap/context_map` — the precompressed sibling context map, served
  verbatim with `Content-Encoding: br` (the file on disk is already
  in transit-ready form).
"""
import io
import os
import threading

import brotli
import large_image
import numpy as np
from flask import Blueprint, request, send_file, abort, jsonify, Response
from large_image.constants import TILE_FORMAT_NUMPY

from previews.create_tile_pyramid import context_map_path_for
from web_server.backend.router import stateful_route
from web_server.backend._route import tap_route


tiles_bp = Blueprint("tiles", __name__)

# Brotli quality for per-tile compression on the fly. Quality 4 gives
# ~30 MB/s throughput on a single core and ~70 % the ratio of quality
# 11 — perfect for the request-time path where latency matters more
# than the last few bytes of size. Quality 11 is reserved for the
# build-time context map precompression.
_TILE_BROTLI_QUALITY = 4


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
    tap_route(
        tiles_bp, "/tap/tile",
        hierarchy=("object_type", "object_id"),
        suffix="/<int:z>/<int:x>/<int:y>.f32",
    ),
    sharding_key="object_id",
)
def serve_tile_tap(
    object_type: str, object_id: str, pipeline: str, display: str, filter: str,
    z: int, x: int, y: int,
):
    path = _tile_pyramid_path_for_tap(object_type, object_id, pipeline, display, filter)
    return _serve_tile_response(path, object_type, object_id, x, y, z)


def _serve_tile_response(path, object_type, object_id, x, y, z):
    """Read one tile out of the pyramid as a numpy float32 array, pack
    as raw little-endian bytes, brotli-compress, return with
    `Content-Encoding: br`. The browser's `fetch` decompresses brotli
    on the network thread; the bytes that arrive in JavaScript are
    direct Float32Array-compatible and can be uploaded to a WebGL
    R32F texture with no parse step.
    """
    if path is None:
        return abort(404)

    try:
        handle = _get_handle(path)
        # Numpy format pulls the raw pixel array out of the pyramid;
        # no PNG/JPEG encoding step (which would have quantized to
        # uint8 and destroyed the linear-float values the shader
        # needs).
        tile = handle.getTile(x, y, z, format=TILE_FORMAT_NUMPY)
        # large_image returns one of three shapes depending on the
        # source pyramid format:
        #   * `np.ndarray`            — the float32 case (linear-float
        #                               TIFFs from the new pyramid
        #                               generator; vips reads them as
        #                               float arrays natively).
        #   * `(np.ndarray, mime)`    — older PIL-source path.
        #   * `ImageBytes` (a `bytes` subclass) — when the source is
        #                               an encoded JPEG/PNG TIFF (the
        #                               legacy autostretched pyramids
        #                               still on disk for some taps).
        # The last case used to crash here because `tile[0]` on bytes
        # is the first *byte* (an int), not a tile array.
        if isinstance(tile, np.ndarray):
            arr = tile
        elif isinstance(tile, tuple) and isinstance(tile[0], np.ndarray):
            arr = tile[0]
        else:
            from PIL import Image
            arr = np.asarray(Image.open(io.BytesIO(bytes(tile))))
        # Multi-channel images: collapse to the first channel for now;
        # the shader is mono. RGB-aware tile serving is a separate
        # change (would require packing 3 channels into the f32 stream
        # and a shader that samples .rgb instead of .r).
        if arr.ndim == 3:
            arr = arr[..., 0]
        # Legacy autostretched JPEG pyramids decode to uint8 0..255;
        # the new shader expects pixels in the same value space as the
        # linear master (≈ 0..1.something), so rescale into [0,1] for
        # uint8 sources. New linear-float pyramids land here already
        # as float32 — passed through untouched.
        if arr.dtype == np.uint8:
            arr = arr.astype("<f4") / 255.0
        else:
            arr = arr.astype("<f4")
        arr = np.ascontiguousarray(arr)
        raw = arr.tobytes()
        compressed = brotli.compress(raw, quality=_TILE_BROTLI_QUALITY)
        resp = Response(compressed, mimetype="application/octet-stream")
        resp.headers["Content-Encoding"] = "br"
        # The shape is implied by the tile-pyramid metadata (the
        # tileWidth/tileHeight the client already has). Include it
        # anyway for debug-friendly out-of-band verification.
        h, w = arr.shape[:2]
        resp.headers["X-Tile-Width"] = str(w)
        resp.headers["X-Tile-Height"] = str(h)
        return resp
    except Exception as e:
        print(f"Error serving tile for {object_type, object_id}: {e}")
        return abort(500)


@stateful_route(
    tap_route(
        tiles_bp, "/tap/context_map",
        hierarchy=("object_type", "object_id"),
    ),
    sharding_key="object_id",
)
def serve_context_map_tap(
    object_type: str, object_id: str, pipeline: str, display: str, filter: str,
):
    """Serve the per-master context map as precompressed brotli.

    The file on disk is already in transit-ready form: brotli-
    compressed binary blob whose decompressed payload is the magic
    header + dimensions + float16 pixels. We `sendfile()` it verbatim
    with `Content-Encoding: br` so the browser decompresses on the
    fetch path without us paying any per-request CPU.

    404 when the source was too small to produce a context map (the
    shader falls back to sampling tile mipmaps at no quality cost for
    that size).
    """
    pyramid_path = _tile_pyramid_path_for_tap(
        object_type, object_id, pipeline, display, filter,
    )
    if pyramid_path is None:
        return abort(404)
    from pathlib import Path
    context_path = context_map_path_for(Path(pyramid_path))
    if not context_path.exists():
        return abort(404)
    resp = send_file(
        str(context_path), mimetype="application/octet-stream",
        conditional=True,  # ETag + Range support for caching
    )
    resp.headers["Content-Encoding"] = "br"
    resp.headers["Cache-Control"] = "private, max-age=3600"
    return resp
