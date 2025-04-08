import io
import os
import time
import threading
import large_image
from pathlib import Path
import numpy as np
import tifffile
from flask import Blueprint, send_file, abort, jsonify

from context import context
from stacks.artifacts import output_filename as stack_output_filename
from mosaics.artifacts import tile_filename as mosaic_output_filename
from file_io.generic import read_image
from previews import autostretch

# Settings
TILE_CACHE_DIR = Path("/tmp/astroflow_tiles")
EXPIRATION_SECONDS = 300  # 5 minutes
TILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Global cache state
_image_cache = {}
_cache_lock = threading.Lock()

tiles_bp = Blueprint("tiles", __name__)


def convert_to_tiff(path: Path, temp_path: Path):
  """Read image and save to .tif"""
  arr = autostretch(read_image(path))
  arr = (arr * 255.0).astype(np.uint8)
  tifffile.imwrite(temp_path, arr, tile=(256, 256), compression='deflate')


def get_image(object_type: str, object_id: str):
  """Get a large_image handle from the cached TIFF, generating it if needed."""
  now = time.time()
  with _cache_lock:
    # Clean up expired entries
    expired = [
        key for key, entry in _image_cache.items()
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
    key = (object_type, object_id)
    if key in _image_cache:
      _image_cache[key]["last_used"] = now
      return _image_cache[key]["handle"]

    # TODO: Use post-processed stack image?
    if object_type == 'stack':
      path = stack_output_filename(context.stacks[object_id])
    elif object_type == 'mosaic':
      path = mosaic_output_filename(context.mosaics[object_id], filter='H')

    path = os.fspath(path)
    if not os.path.exists(path):
      print(f"Path not found: {path}")
      return None

    print(f"Opening file {path} with large_image...")
    handle = large_image.open(path, plugin='vips')
    _image_cache[key] = {
        "handle": handle,
        "path": path,
        "last_used": now,
    }
    return handle


@tiles_bp.route('/tile_metadata/<object_type>/<object_id>')
def tile_metadata(object_type, object_id):
  source = get_image(object_type, object_id)
  meta = source.getMetadata()
  return jsonify({
      'width': meta['sizeX'],
      'height': meta['sizeY'],
      'tileWidth': meta['tileWidth'],
      'tileHeight': meta['tileHeight'],
      'maxZoom': meta['levels'] - 1
  })


@tiles_bp.route("/tile/<object_type>/<object_id>/<int:z>/<int:x>/<int:y>.png")
def serve_tile(object_type, object_id, z, x, y):
  source = get_image(object_type, object_id)
  if source is None:
    return abort(404)

  try:
    tile_data = source.getTile(x, y, z, encoding='PNG')
    return send_file(io.BytesIO(tile_data), mimetype="image/png")
  except Exception as e:
    print(f"Error serving tile for {object_type, object_id}: {e}")
    return abort(500)
