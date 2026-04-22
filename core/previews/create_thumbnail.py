import os
import numpy as np
from functools import wraps
from typing import Callable, Optional
from pathlib import Path
import tempfile

from file_io.generic import read_image
from previews.operations import resize_array, autostretch, array_to_png
from protocols import CacheProtocol


def default_thumbnail_fn(path: Path, width=2048) -> Optional[np.ndarray]:
    return array_to_png(
        autostretch(np.clip(resize_array(read_image(path), width), 0.0, 1.0))
    )


def make_create_thumbnail_decorator(
    cache: CacheProtocol,
    thumbnail_fn: Callable[[Path], Optional[np.ndarray]] = default_thumbnail_fn,
):
    def create_thumbnail(func: Callable = None, *, lookup_args: Callable = None):
        """Decorator that caches a thumbnail after the wrapped function writes an image.

        Thumbnails are keyed by the output filename so that the web
        server can look them up via ``filename_fn(*args)`` from the
        ``@makes_image`` registry — no arg-hash matching needed.

        When ``lookup_args`` is provided, its return value is used to
        compute a custom output filename for storage/lookup. This lets
        multiple calls (e.g. progressive mosaic merges) overwrite the
        same thumbnail key.
        """
        if func is not None:
            return _wrap(func, lookup_args=None)
        return lambda f: _wrap(f, lookup_args=lookup_args)

    def _wrap(func: Callable, lookup_args: Callable = None):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            if result is None:
                return result

            if isinstance(result, tempfile._TemporaryFileWrapper):
                filename = result.name
            else:
                filename = os.fspath(result)

            if os.path.exists(filename):
                if thumbnail := thumbnail_fn(filename):
                    thumbnail_key = ("thumbnail", str(filename))
                    cache[thumbnail_key] = thumbnail
                    print(
                        f"### Wrote thumbnail_key {thumbnail_key} "
                        f"for filename {filename}"
                    )

            return result

        def get_thumbnail_fn(*args, **kwargs) -> Optional[np.ndarray]:
            if hasattr(wrapper, 'filename_fn'):
                try:
                    path = wrapper.filename_fn(*args, **kwargs)
                    if path:
                        return cache.get(("thumbnail", str(path)))
                except (TypeError, KeyError):
                    pass
            return None

        wrapper.thumbnail_fn = get_thumbnail_fn

        return wrapper

    return create_thumbnail
