import os
import numpy as np
from functools import wraps
from typing import Callable, Optional
from pathlib import Path
import tempfile

from hashing import hash_call, hash_source_code
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
    def create_thumbnail(
        cache: CacheProtocol,
        thumbnail_fn: Callable[[Path], Optional[np.ndarray]],
    ):
        def decorator(func: Callable):
            fn_hash = hash_source_code(func)

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
                        call_hash = hash_call(fn_hash, *args, **kwargs)
                        thumbnail_key = ("thumbnail", call_hash)
                        cache[thumbnail_key] = thumbnail
                        print(
                            f"### Wrote thumbnail_key {thumbnail_key} "
                            f"for filename {filename}"
                        )

                        # Store an index by first arg so thumbnail_fn(first_arg)
                        # can find thumbnails for functions with multiple args.
                        if args:
                            first_arg_hash = hash_call(fn_hash, args[0])
                            cache[("thumbnail_index", first_arg_hash)] = call_hash

                return result

            def get_thumbnail_fn(*args, **kwargs) -> Optional[np.ndarray]:
                # Try index lookup by first arg (works for single-arg calls).
                if args:
                    first_arg_hash = hash_call(fn_hash, args[0])
                    index_key = ("thumbnail_index", first_arg_hash)
                    if (call_hash := cache.get(index_key)) is not None:
                        return cache.get(("thumbnail", call_hash))

                # Fall back to full-args lookup.
                call_hash = hash_call(fn_hash, *args, **kwargs)
                return cache.get(("thumbnail", call_hash))

            wrapper.thumbnail_fn = get_thumbnail_fn

            return wrapper

        return decorator

    return create_thumbnail(cache, thumbnail_fn)
