"""@create_tile_pyramid: a side-attribute decorator for image producers.

Mirrors the structure of @create_thumbnail. Wraps an image producer so
that, immediately after the producer writes its main output, a tiled
TIFF "tile pyramid" derivative is also written next to it on disk. The
tile pyramid path is indexed in the cache (twice — once by the
producer's full call_hash and once by an index built from just the
first argument), and the wrapped producer gains a `tile_pyramid_fn`
attribute that consumers (e.g. the web server) can call to look up the
tile pyramid for a given object without knowing the producer's full
argument signature.

Cache key shape:
    ("tile_pyramid", call_hash)        -> str(path)
    ("tile_pyramid_index", first_arg_hash) -> call_hash

Both keys are namespaced under the producer's `fn_hash`, so different
producers don't collide. The full-args path is the canonical lookup;
the first-arg index exists so the web server can call
    producer.tile_pyramid_fn(stack)
without having to construct a full ProcessingConfig.
"""
import os
import tempfile
from functools import wraps
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import tifffile

from hashing import hash_source_code
from file_io.generic import read_image
from previews.operations import autostretch
from protocols import CacheProtocol


def default_make_tile_pyramid(source_path: Path, dest_path: Path) -> None:
    """Read an image and write a tiled TIFF suitable for serving via large_image.

    Autostretches the source, scales to uint8, writes a tiled TIFF
    with 256x256 tiles and deflate compression. The output is not
    multi-resolution (no pyramid overviews) — large_image / vips can
    still serve it cleanly, computing downsampled tiles on the fly.
    """
    arr = autostretch(read_image(source_path))
    arr = (arr * 255.0).astype(np.uint8)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(dest_path, arr, tile=(256, 256), compression="deflate")


def make_create_tile_pyramid_decorator(
    cache: CacheProtocol,
    make_tile_pyramid: Callable[[Path, Path], None] = default_make_tile_pyramid,
):
    def create_tile_pyramid(func: Callable = None, *, tile_filename: Callable = None):
        """Decorator that generates a tile pyramid after the wrapped
        function produces an image.

        Usage::

            @create_tile_pyramid                          # default: .tiff next to source
            def my_producer(...): ...

            @create_tile_pyramid(tile_filename=my_fn)     # custom: my_fn(*args, **kwargs) -> Path
            def my_producer(...): ...

        When ``tile_filename`` is provided, the tile pyramid is written
        to ``tile_filename(*args, **kwargs)`` (overwriting any previous
        file at that path) instead of being derived from the source.
        This is useful when multiple calls should overwrite a single
        tile pyramid at a fixed location (e.g. in-progress mosaic
        merges).
        """
        if func is not None:
            # Bare @create_tile_pyramid — func is the decorated function.
            return _wrap(func, tile_filename=None)
        # @create_tile_pyramid(tile_filename=...) — return a decorator.
        return lambda f: _wrap(f, tile_filename=tile_filename)

    def _wrap(func: Callable, tile_filename: Callable = None):
        fn_hash = hash_source_code(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if result is None:
                return result

            # Resolve the producer's output to a real path on disk.
            if isinstance(result, tempfile._TemporaryFileWrapper):
                filename = result.name
            else:
                filename = os.fspath(result)

            if not os.path.exists(filename):
                return result

            source = Path(filename)

            if tile_filename is not None:
                # Custom destination — always write a fresh tile pyramid
                # (overwriting any previous file at this path).
                tile_pyramid_path = Path(tile_filename(*args, **kwargs))
                tile_pyramid_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    make_tile_pyramid(source, tile_pyramid_path)
                except Exception as e:
                    print(
                        f"### Failed to make tile pyramid for {source}: {e}"
                    )
                    return result
            elif source.suffix.lower() in (".tif", ".tiff"):
                # Source is already a TIFF (e.g. subframe-mode mosaics,
                # which write a tiled tiff directly) — use it as its
                # own tile pyramid, no extra file needed.
                tile_pyramid_path = source
            else:
                tile_pyramid_path = source.with_suffix(".tiff")
                try:
                    make_tile_pyramid(source, tile_pyramid_path)
                except Exception as e:
                    print(
                        f"### Failed to make tile pyramid for {source}: {e}"
                    )
                    return result

            cache[("tile_pyramid", str(source))] = str(tile_pyramid_path)

            print(
                f"### Wrote tile pyramid {tile_pyramid_path.name} "
                f"for producer output {source.name}"
            )

            return result

        def get_tile_pyramid_fn(*args, **kwargs) -> Optional[Path]:
            if hasattr(wrapper, 'filename_fn'):
                try:
                    output_path = wrapper.filename_fn(*args, **kwargs)
                    if output_path:
                        path = cache.get(("tile_pyramid", str(output_path)))
                        if path is not None:
                            return Path(path)
                except (TypeError, KeyError):
                    pass
            return None

        wrapper.tile_pyramid_fn = get_tile_pyramid_fn
        return wrapper

    return create_tile_pyramid
