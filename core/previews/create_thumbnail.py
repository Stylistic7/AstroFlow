import numpy as np
from functools import wraps
from typing import Callable, Optional
from pathlib import Path

from hashing import hash_call, hash_source_code
from file_io.generic import read_image
from previews.operations import resize_array, autostretch
from protocols import CacheProtocol


def default_thumbnail_fn(path: Path, width=2048) -> Optional[np.ndarray]:
  try:
    image = read_image(path)
    return autostretch(np.clip(resize_array(image, width), 0.0, 1.0))
  except Exception as e:
    return None


def create_thumbnail(
    cache: CacheProtocol,
    thumbnail_fn: Callable[[Path],
                           Optional[np.ndarray]] = default_thumbnail_fn):

  def decorator(func: Callable):
    fn_hash = hash_source_code(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
      result = func(*args, **kwargs)
      if isinstance(result, Path):
        thumbnail = thumbnail_fn(result)
        if thumbnail is not None:
          call_hash = hash_call(fn_hash, *args, **kwargs)
          thumbnail_key = ('thumbnail', call_hash)
          cache[thumbnail_key] = thumbnail

      return result

    def get_thumbnail_fn(*args, **kwargs) -> Optional[np.ndarray]:
      call_hash = hash_call(fn_hash, *args, **kwargs)
      thumbnail_key = ('thumbnail', call_hash)
      return cache.get(thumbnail_key)

    wrapper.thumbnail_fn = get_thumbnail_fn

    return wrapper

  return decorator
