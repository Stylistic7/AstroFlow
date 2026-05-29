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
import pyvips

from hashing import hash_source_code
from file_io.generic import read_image
from previews.operations import autostretch
from protocols import CacheProtocol


def default_make_tile_pyramid(source_path: Path, dest_path: Path) -> None:
    """Read an image and write a *pyramidal* tiled TIFF for serving via
    large_image / pyvips.

    Autostretches the source to uint8, then writes a tiled,
    multi-resolution TIFF: 256x256 tiles, deflate compression, and
    factor-2 overview levels stored as extra IFDs. large_image reads
    the overviews as zoom levels, and the coarsest level doubles as a
    fast-read thumbnail — so neither a low-zoom tile nor a thumbnail
    needs to decode the full-resolution image.
    """
    arr = np.ascontiguousarray((autostretch(read_image(source_path)) * 255.0).astype(np.uint8))
    height, width = arr.shape[:2]
    bands = 1 if arr.ndim == 2 else arr.shape[2]
    image = pyvips.Image.new_from_memory(arr.tobytes(), width, height, bands, "uchar")
    tmp = dest_path.with_suffix(dest_path.suffix + ".tmp")
    image.tiffsave(
        str(tmp), tile=True, pyramid=True, compression="deflate",
        tile_width=256, tile_height=256, bigtiff=True,
    )
    os.replace(tmp, dest_path)


_THUMBNAIL_MIN_PX = 1200


def pyramid_thumbnail(tile_pyramid_path, min_px: int = _THUMBNAIL_MIN_PX) -> bytes:
    """PNG bytes of the smallest stored pyramid level whose min dimension
    is >= `min_px`, read verbatim — the browser scales it for display, so
    there is no server-side resize. Reads only that one level (a few MB),
    never the full-resolution image.

    For a legacy single-resolution `.tiff` (no overviews) it shrink-on-
    loads via libvips instead, so it still never materializes full res."""
    path = os.fspath(tile_pyramid_path)
    base = pyvips.Image.tiffload(path, page=0)
    try:
        n_pages = base.get("n-pages")
    except pyvips.Error:
        n_pages = 1
    if n_pages > 1:
        # Pyramid levels are factor-2 (page 0 = full res). Pick the
        # deepest level still >= min_px on its short edge.
        level, dim = 0, min(base.width, base.height)
        while level + 1 < n_pages and dim // 2 >= min_px:
            level += 1
            dim //= 2
        image = pyvips.Image.tiffload(path, page=level)
    else:
        image = pyvips.Image.thumbnail(path, min_px)
    return image.write_to_buffer(".png")


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

        def get_tile_pyramid_fn(*args, level=None, **kwargs) -> Optional[Path]:
            if hasattr(wrapper, 'filename_fns'):
                for (obj_type, lvl), fn in wrapper.filename_fns.items():
                    if level is not None and lvl != level:
                        continue
                    try:
                        output_path = fn(*args, **kwargs)
                        if output_path:
                            path = cache.get(("tile_pyramid", str(output_path)))
                            if path is not None:
                                return Path(path)
                    except (TypeError, KeyError):
                        continue
            return None

        wrapper.tile_pyramid_fn = get_tile_pyramid_fn
        return wrapper

    return create_tile_pyramid
