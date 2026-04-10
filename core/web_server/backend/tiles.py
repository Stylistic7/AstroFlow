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
from stacks.composite_stack import Stack
from stacks.composite_stack import CalibrationStack
from targets.types import Target
from mosaics.types import Mosaic
from products.registry import get_producer, get_filename_fn, MediaType
from web_server.backend.router import stateful_route

# Settings
TILE_CACHE_DIR = Path("/tmp/astroflow_tiles")
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


def get_image_for_level(object_type: type, level: QualityLevel, *args) -> Path | None:
    """Get the image path for a specific quality level."""
    filename_fn = get_filename_fn(MediaType.IMAGE, object_type, level)
    if filename_fn is None:
        return None
    fragment = filename_fn(*args)
    if fragment and os.path.exists(fragment.resolve_at(context.output_dir)):
        return fragment
    return None


def get_image(
    object_type: str, object_id: str, filter: str, level: str = None, load_if_missing: bool = False,
):
    """Get a large_image handle from the cached TIFF, generating it if needed."""
    now = time.time()
    with _cache_lock:
        key = (object_type, object_id, filter, level)
        if key in _image_cache:
            entry = _image_cache[key]
            # Evict if the source TIFF was modified since we cached it.
            tiff_path = entry.get("tiff_path")
            if tiff_path and os.path.exists(tiff_path):
                disk_mtime = os.path.getmtime(tiff_path)
                if disk_mtime > entry.get("mtime", 0):
                    print(f"Tile cache stale for {key}, reloading")
                    del _image_cache[key]
                else:
                    entry["last_used"] = now
                    return entry["handle"]
            else:
                entry["last_used"] = now
                return entry["handle"]

        if not load_if_missing:
            return None

        # Resolve the object and find the image at the requested level.
        type_map = {"stack": Stack, "calibration_stack": CalibrationStack, "target": Target, "mosaic": Mosaic}
        obj_cls = type_map.get(object_type)
        if obj_cls is None:
            raise ValueError(f"Unknown object_type {object_type}")

        if object_type == "stack":
            obj_args = (context.stacks[object_id].get(),)
        elif object_type == "calibration_stack":
            obj_args = (context.calibration_stacks[object_id],)
        elif object_type == "target":
            obj_args = (context.targets[object_id].get(), filter)
        elif object_type == "mosaic":
            obj_args = (context.mosaics[object_id].get(), filter)

        if level and level != "auto":
            ql = QualityLevel(level)
            path = get_image_for_level(obj_cls, ql, *obj_args)
        else:
            # Fallback: find best available.
            path = None
            for ql in [QualityLevel.POST_PROCESSED, QualityLevel.MASTER, QualityLevel.DOWNSCALED_MASTER, QualityLevel.SUBFRAME]:
                path = get_image_for_level(obj_cls, ql, *obj_args)
                if path:
                    break

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
            "mtime": os.path.getmtime(tiff_path),
            "last_used": now,
        }
        return handle


@stateful_route(
    tiles_bp.route("/tile_metadata/<object_type>/<object_id>/<filter>/<level>"),
    sharding_key="object_id",
)
def tile_metadata(
    object_type: str, object_id: str, filter: str, level: str = "auto", backend_port: int = None,
):
    host_port = request.host.split(":")[-1]
    print(f"### tile_metadata calling get_image on worker on port {host_port} level={level}")
    source = get_image(object_type, object_id, filter, level=level, load_if_missing=True)

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
    "/tile/<object_type>/<object_id>/<filter>/<level>/<int:z>/<int:x>/<int:y>.png"
)
@cross_origin(origins="*")
def serve_tile(
    object_type: str, object_id: str, filter: str, level: str, z: str, x: str, y: str,
):
    source = get_image(object_type, object_id, filter, level=level, load_if_missing=False)

    if source in [None, IN_PROGRESS]:
        return Response(status=202, headers={"Retry-After": "5"})

    try:
        tile_data = source.getTile(x, y, z, encoding="PNG")
        return send_file(io.BytesIO(tile_data), mimetype="image/png")
    except Exception as e:
        print(f"Error serving tile for {object_type, object_id}: {e}")
        return abort(500)
