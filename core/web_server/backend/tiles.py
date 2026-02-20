import io
import os
import time
import threading
import large_image
from pathlib import Path
import numpy as np
import tifffile
from flask import Blueprint, Response, send_file, abort, jsonify, request
from flask_cors import cross_origin

from context import context
from file_io.generic import read_image
from file_io.types import TiffFilePath, WithSidecar
from previews.operations import autostretch
from previews.types import QualityLevel
from stacks.composite_stack import CompositeStack
from stacks.simple_stack import CalibrationStack
from targets.types import Target
from mosaics.types import Mosaic
from products.registry import get_producer, MediaType
from web_server.backend.router import stateful_route

# Settings
TILE_CACHE_DIR = Path("/tmp/astroflow_tiles")
EXPIRATION_SECONDS = 300  # 5 minutes
TILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
IN_PROGRESS = object()

# Global cache state
_image_cache = {}
_cache_lock = threading.Lock()

tiles_bp = Blueprint("tiles", __name__)


def convert_to_tiff(source_path: Path, dest_path: Path):
    """Read image and save to .tif"""
    print(f"Converting {source_path} to {dest_path}...")
    assert source_path != dest_path
    arr = autostretch(read_image(source_path))
    arr = (arr * 255.0).astype(np.uint8)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(dest_path, arr, tile=(256, 256), compression="deflate")
    print(f"Conversion done.")


def get_best_image(object_type: type, *args) -> Path:
    for level in [
        QualityLevel.POST_PROCESSED,
        QualityLevel.MASTER,
        QualityLevel.SUBFRAME,
    ]:
        if fn := get_producer(MediaType.IMAGE, object_type, level):
            if fragment := fn.filename_fn(*args):
                filename = fragment.resolve_at(context.output_dir)
                if os.path.exists(filename):
                    return fragment


def get_image(
    object_type: str, object_id: str, filter: str, load_if_missing: bool = False
):
    """Get a large_image handle from the cached TIFF, generating it if needed."""
    now = time.time()
    with _cache_lock:
        # Clean up expired entries
        expired = [
            key
            for key, entry in _image_cache.items()
            if now - entry["last_used"] > EXPIRATION_SECONDS
        ]
        for key in expired:
            print(f"Unloading and deleting cached image: {key}")
            entry = _image_cache.pop(key)
            try:
                entry["handle"].close()
            except Exception as e:
                print(f"Error closing handle: {e}")
            try:
                entry["temp_path"].unlink()
            except Exception as e:
                print(f"Error deleting temp file: {e}")

        # Reuse if already in cache
        key = (object_type, object_id, filter)
        if key in _image_cache:
            _image_cache[key]["last_used"] = now
            return _image_cache[key]["handle"]

        if not load_if_missing:
            return None

        # Else, convert from .xisf and cache
        if object_type == "stack":
            path = get_best_image(CompositeStack, context.stacks[object_id])
        elif object_type == "calibration_stack":
            path = get_best_image(
                CalibrationStack, context.calibration_stacks[object_id]
            )
        elif object_type == "target":
            path = get_best_image(Target, context.targets[object_id], filter)
        elif object_type == "mosaic":
            path = get_best_image(Mosaic, context.mosaics[object_id], filter)
        else:
            raise ValueError(f"Unknown object_type {object_type}")

        if path is None:
            print(f"No image found")
            return None

        # This prevents multiple requests from running the same image conversion
        # multiple times in parallel.
        _image_cache[key] = {
            "handle": IN_PROGRESS,
            "last_used": now,
        }

        if isinstance(path, WithSidecar):
            path = path.path

        if isinstance(path, TiffFilePath):
            tiff_path = path.resolve_at(context.output_dir)
        else:
            if path.is_absolute():
                tiff_path = TILE_CACHE_DIR / path.with_suffix(".tiff").name
                print(f"### absolute path {path=} {tiff_path=}")
                convert_to_tiff(path, tiff_path)
            else:
                original_path = path.resolve_at(context.output_dir)
                tiff_path = path.resolve_at(TILE_CACHE_DIR).with_suffix(".tiff")
                print(
                    f"### relative path {path=} {original_path=} {tiff_path=}"
                )
                convert_to_tiff(original_path, tiff_path)

        print(f"Opening file {tiff_path} with large_image...")
        handle = large_image.open(tiff_path, plugin="vips")
        _image_cache[key] = {
            "handle": handle,
            "tiff_path": tiff_path,
            "last_used": now,
        }
        return handle


@stateful_route(
    tiles_bp.route("/tile_metadata/<object_type>/<object_id>/<filter>"),
    sharding_key="object_id",
)
def tile_metadata(
    object_type: str, object_id: str, filter: str, backend_port: int
):
    host_port = request.host.split(":")[-1]
    print(f"### tile_metadata calling get_image on worker on port {host_port}")
    source = get_image(object_type, object_id, filter, load_if_missing=True)

    if source is None:
        return abort(404)

    if source is IN_PROGRESS:
        return Response(status=202, headers={"Retry-After": "5"})

    meta = source.getMetadata()
    return jsonify(
        {
            "width": meta["sizeX"],
            "height": meta["sizeY"],
            "tileWidth": meta["tileWidth"],
            "tileHeight": meta["tileHeight"],
            "maxZoom": meta["levels"] - 1,
            "port": backend_port,
        }
    )


@tiles_bp.route(
    "/tile/<object_type>/<object_id>/<filter>/<int:z>/<int:x>/<int:y>.png"
)
@cross_origin(origins="*")
def serve_tile(
    object_type: str, object_id: str, filter: str, z: str, x: str, y: str
):
    source = get_image(object_type, object_id, filter, load_if_missing=False)

    if source in [None, IN_PROGRESS]:
        return Response(status=202, headers={"Retry-After": "5"})

    try:
        tile_data = source.getTile(x, y, z, encoding="PNG")
        return send_file(io.BytesIO(tile_data), mimetype="image/png")
    except Exception as e:
        print(f"Error serving tile for {object_type, object_id}: {e}")
        return abort(500)
