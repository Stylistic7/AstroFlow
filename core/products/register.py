import functools
from pathlib import Path
from PIL import Image
from typing import Callable

from context import context
from previews.operations import resize_array, array_to_png
from file_io.generic import read_image


def register_image_producer(object_type: str, level: str, tiled: bool = False):

  def decorator(func: Callable[..., Path]):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      image_path: Path = func(*args, **kwargs)

      thumb_key = ('thumbnail', object_type, level, args[0])
      if thumb_key not in context:
        thumbnail = _generate_thumbnail(image_path)
        context.set(thumb_key, thumbnail)

      # Register metadata
      metadata_key = ('metadata', object_type, args[0])
      existing = context.get(metadata_key, {})
      existing[level] = {
          'type': 'tiled' if tiled else 'png',
          'image_url': f'/image/{object_type}/{level}/{args[0]}',
          'thumbnail_url': f'/thumbnail/{object_type}/{level}/{args[0]}',
          'created_at': image_path.stat().st_mtime,
          #   'size': _get_image_size(image_path),
      }
      context.set(metadata_key, existing)

      return image_path

    return wrapper

  return decorator


def _generate_thumbnail(image_path: Path) -> bytes:
  return array_to_png(resize_array(read_image(image_path), 1500))
