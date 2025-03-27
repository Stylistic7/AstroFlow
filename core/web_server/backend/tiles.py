import io
import os
import time
import threading
import large_image
from pathlib import Path
import numpy as np
import tifffile
from PIL import Image
from flask import Blueprint, send_file, abort, jsonify

from context import context
from stacks.artifacts import output_filename
from stacks.types import StackHash
from file_io.xisf_file_io import read_xisf_image
from previews import autostretch

# Settings
TILE_CACHE_DIR = Path("/tmp/astroflow_tiles")
EXPIRATION_SECONDS = 300  # 5 minutes
TILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Global cache state
_image_cache = {}
_cache_lock = threading.Lock()

tiles_bp = Blueprint("tiles", __name__)


def convert_xisf_to_tiff(xisf_path: Path, temp_path: Path):
  """Read .xisf image and save to .tif"""
  arr = autostretch(read_xisf_image(xisf_path))
  arr = (arr * 255.0).astype(np.uint8)
  tifffile.imwrite(temp_path,
                   arr,
                   dtype=arr.dtype,
                   tile=(256, 256),
                   bigtiff=True,
                   compression='deflate')


def get_cached_image(stack_id: StackHash):
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
    if stack_id in _image_cache:
      _image_cache[stack_id]["last_used"] = now
      return _image_cache[stack_id]["handle"]

    # Else, convert from .xisf and cache
    xisf_path = output_filename(context.stacks[stack_id])
    if not xisf_path.exists():
      print(f"XISF not found: {xisf_path}")
      return None

    temp_path = TILE_CACHE_DIR / f"{stack_id}.tif"
    if not temp_path.exists():
      print(f"Converting {xisf_path} to {temp_path}...")
      convert_xisf_to_tiff(xisf_path, temp_path)

    print(f"Opening file {temp_path} with large_image...")
    handle = large_image.open(temp_path, plugin='vips')
    _image_cache[stack_id] = {
        "handle": handle,
        "temp_path": temp_path,
        "last_used": now,
    }
    return handle


@tiles_bp.route('/tile_metadata/<image_hash>')
def tile_metadata(image_hash):
  source = get_cached_image(image_hash)
  meta = source.getMetadata()
  return jsonify({
      'width': meta['sizeX'],
      'height': meta['sizeY'],
      'tileWidth': meta['tileWidth'],
      'tileHeight': meta['tileHeight'],
      'maxZoom': meta['levels'] - 1
  })


@tiles_bp.route("/tiles/<stack_id>/<int:z>/<int:x>/<int:y>.png")
def serve_tile(stack_id, z, x, y):
  source = get_cached_image(stack_id)
  if source is None:
    return abort(404)

  try:
    tile_data = source.getTile(x, y, z, encoding='PNG')
    return send_file(io.BytesIO(tile_data), mimetype="image/png")
  except Exception as e:
    print(f"Error serving tile for {stack_id}: {e}")
    return abort(500)
