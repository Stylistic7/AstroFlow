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

from hashing import hash_call, hash_source_code
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
    def create_tile_pyramid(func: Callable):
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

            # If the producer's main output is already a TIFF (e.g.
            # subframe-mode mosaics, which write a tiled tiff directly),
            # use it as its own tile pyramid — no extra file needed.
            if source.suffix.lower() in (".tif", ".tiff"):
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

            # Index the tile pyramid path by the producer's call_hash.
            call_hash = hash_call(fn_hash, *args, **kwargs)
            cache[("tile_pyramid", call_hash)] = str(tile_pyramid_path)

            # Index by first arg too, so consumers can look up with
            # partial args (e.g. just the stack/mosaic without config).
            if args:
                first_arg_hash = hash_call(fn_hash, args[0])
                cache[("tile_pyramid_index", first_arg_hash)] = call_hash

            print(
                f"### Wrote tile pyramid {tile_pyramid_path.name} "
                f"for producer output {source.name}"
            )

            return result

        def get_tile_pyramid_fn(*args, **kwargs) -> Optional[Path]:
            # Try index lookup by first arg first (this is the path
            # consumers like the web server actually use).
            if args:
                first_arg_hash = hash_call(fn_hash, args[0])
                index_key = ("tile_pyramid_index", first_arg_hash)
                if (call_hash := cache.get(index_key)) is not None:
                    path = cache.get(("tile_pyramid", call_hash))
                    if path is not None:
                        return Path(path)

            # Fall back to full-args lookup.
            call_hash = hash_call(fn_hash, *args, **kwargs)
            path = cache.get(("tile_pyramid", call_hash))
            return Path(path) if path else None

        wrapper.tile_pyramid_fn = get_tile_pyramid_fn
        return wrapper

    return create_tile_pyramid
