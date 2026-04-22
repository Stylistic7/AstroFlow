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
        server can look them up via the level-specific ``filename_fns``
        dict from ``@makes_image`` — no arg-hash matching needed.

        When ``lookup_args`` is provided, its return value is used as
        the cache key instead. This lets multiple calls (e.g.
        progressive mosaic merges) overwrite the same thumbnail key.
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
                    if lookup_args is not None:
                        try:
                            canonical = str(lookup_args(*args, **kwargs))
                        except (TypeError, KeyError):
                            canonical = str(filename)
                    else:
                        canonical = str(filename)
                    thumbnail_key = ("thumbnail", canonical)
                    cache[thumbnail_key] = thumbnail
                    print(
                        f"### Wrote thumbnail_key {thumbnail_key} "
                        f"for filename {filename}"
                    )

            return result

        def get_thumbnail_fn(*args, level=None, **kwargs) -> Optional[np.ndarray]:
            if lookup_args is not None:
                try:
                    if level is not None:
                        key = str(lookup_args(level, *args, **kwargs))
                    else:
                        key = str(lookup_args(*args, **kwargs))
                    return cache.get(("thumbnail", key))
                except (TypeError, KeyError):
                    pass
            if hasattr(wrapper, 'filename_fns'):
                for (obj_type, lvl), fn in wrapper.filename_fns.items():
                    if level is not None and lvl != level:
                        continue
                    try:
                        path = fn(*args, **kwargs)
                        if path:
                            result = cache.get(("thumbnail", str(path)))
                            if result:
                                return result
                    except (TypeError, KeyError):
                        continue
            return None

        wrapper.thumbnail_fn = get_thumbnail_fn

        return wrapper

    return create_thumbnail
