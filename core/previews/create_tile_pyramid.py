"""@create_tile_pyramid: a side-attribute decorator for image producers.

Mirrors the structure of @create_thumbnail. Wraps an image producer so
that, immediately after the producer writes its main output, a
*pyramidal* tiled TIFF is also written next to it on disk. The pixel
data inside the pyramid is the producer's LINEAR pixel data — no
display-time autostretch, no uint8 quantization — because the web
client renders each pixel through a tone-mapping shader at display
time. The shader needs the original floating-point values so that
stretch, structure-emphasis, and HDR brightness can all be re-derived
without any precision loss.

A sibling **context map** is also written. It's a single
Lanczos3-downsampled copy of the linear master at exactly 1:5 ratio,
stored as float16 + brotli-precompressed for fast network transit.
The context map is the texture the WebGL shader binds for local
tone-mapping adaptation; auto-mipmap generation on the GPU gives the
shader multi-scale local context out of one bound texture.

Cache key shape (unchanged from the previous version):
    ("tile_pyramid", call_hash)             -> str(path)
    ("tile_pyramid_index", first_arg_hash)  -> call_hash
Plus, additively, for the context map:
    ("context_map", str(source))            -> str(path)

Both keys are namespaced under the producer's `fn_hash`, so different
producers don't collide.
"""
import os
import struct
import tempfile
from functools import wraps
from pathlib import Path
from typing import Callable, Optional

import brotli
import numpy as np
import pyvips

from hashing import hash_source_code
from file_io.generic import read_image
from protocols import CacheProtocol


# ---------------------------------------------------------------------------
# Tile pyramid: linear float32, deflate-compressed inside the TIFF, brotli
# compression handled at the HTTP transit layer (not at rest).
# ---------------------------------------------------------------------------

_TILE_SIZE = 256


def default_make_tile_pyramid(source_path: Path, dest_path: Path) -> None:
    """Read an image and write a *pyramidal* tiled TIFF whose pixel
    data is the source's LINEAR float32 values (no autostretch, no
    uint8 quantization).

    Compression is TIFF-internal deflate — pyvips decompresses
    on-the-fly when serving individual tiles, and HTTP transit applies
    brotli on top of the raw float32 bytes. The two compression layers
    address different problems: deflate keeps the on-disk file
    manageable; brotli (added at serve time) shrinks the per-tile
    network payload.

    Pyramid levels are factor-2; the default downsample inside
    `tiffsave(pyramid=True)` is libvips' fast block-average shrink.
    For intermediate-zoom DISPLAY purposes that's fine — modern
    monitor resolution and the WebGL bilinear texture sampler soften
    the result enough that the difference from a Lanczos shrink is
    not perceptible at any zoom level the user actually views. The
    Lanczos3 quality goes into the CONTEXT MAP (separate sibling
    artifact, below), where smoothness matters for the local
    tone-mapping operator and would be visible as adaptation
    artifacts under a box-averaged shrink.
    """
    arr = read_image(source_path)
    # Float32 keeps the shader's input as the source's actual linear
    # values; float16 would lose precision for very-bright pixels in
    # high-dynamic-range HDR-display use.
    if arr.ndim == 2:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        bands = 1
    else:
        # Multi-channel — preserve the channel axis as TIFF bands.
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        bands = arr.shape[2]

    height, width = arr.shape[:2]
    # Hand pyvips the numpy buffer directly (zero-copy). `arr.tobytes()`
    # would materialize a SECOND full copy of the array before pyvips even
    # starts, doubling peak memory to ~2x the source — enough to OOM the
    # build on multi-GB masters (a 9 GB FITS peaked at ~18 GB). That OOM is
    # caught by the caller and swallowed, silently leaving a master with no
    # display pyramid (→ 404 in the viewer). pyvips references this buffer
    # and pins it alive on the returned image, so `arr` must outlive
    # `image`; both are locals here and `tiffsave` completes before return.
    image = pyvips.Image.new_from_memory(
        arr.data, width, height, bands, "float",
    )

    tmp = dest_path.with_suffix(dest_path.suffix + ".tmp")
    image.tiffsave(
        str(tmp),
        tile=True, pyramid=True,
        # Deflate is lossless and works well on float data; the
        # alternative (`lzw`) gives slightly larger files. `none`
        # would mean a 4× larger on-disk size, which we'd prefer
        # to avoid even though tiles aren't disk-bound at serve time.
        compression="deflate",
        tile_width=_TILE_SIZE, tile_height=_TILE_SIZE,
        # bigtiff so multi-GB linear-float mosaics save cleanly.
        bigtiff=True,
    )
    os.replace(tmp, dest_path)


# ---------------------------------------------------------------------------
# Context map: a single sibling file. Lanczos3-downsampled at exactly 1:5,
# float16, brotli-precompressed for fast transit. The WebGL shader binds
# this as `uContextMap` for local tone-mapping adaptation; mipmaps generated
# automatically on the GPU give the shader multi-scale local context.
# ---------------------------------------------------------------------------

# 1:5 source-to-context ratio. Chosen so each context pixel represents
# ~5 source pixels — well below the human visual system's CSF-peak
# adaptation scale, which means the local tone-mapping operator has
# enough resolution to produce smooth, halo-free output at any
# aesthetic slider position. Above this ratio (e.g. 1:10), the
# painterly axis loses visible headroom; below (1:2), context-map
# bandwidth gets gratuitous without perceptual benefit.
_CONTEXT_RATIO = 5

# Below this size the source itself fits in a few tiles and the
# shader can sample the tile texture directly with mipmaps for local
# context — no separate context map needed. 256 px (the tile size)
# is the natural threshold.
_MIN_CONTEXT_DIM = 256

# Magic header written at the start of the brotli-decompressed stream:
# 4 bytes ASCII tag + 4-byte LE width + 4-byte LE height. Lets the
# browser parse the layout without a separate metadata fetch.
_CONTEXT_MAGIC = b"CMv1"


def default_make_context_map(source_path: Path, dest_path: Path) -> bool:
    """Lanczos3-downsample the linear source at exactly 1:5 ratio,
    convert to float16, prepend a small dimension header, and brotli-
    precompress. Returns True if a context map was written.

    Returns False (and writes nothing) when the source is small
    enough that source_dim/5 falls below `_MIN_CONTEXT_DIM` on either
    axis — at that point the shader can sample the source tile
    directly with mipmaps, with no quality loss compared to a
    separate map.

    File layout (after brotli decompression):
        offset 0:  magic tag       (4 bytes, b"CMv1")
        offset 4:  width           (4 bytes, little-endian uint32)
        offset 8:  height          (4 bytes, little-endian uint32)
        offset 12: pixel data      (width × height × 2 bytes, float16
                                    row-major)
    """
    image = pyvips.Image.new_from_file(str(source_path), access="sequential")
    w, h = image.width, image.height
    new_w = w // _CONTEXT_RATIO
    new_h = h // _CONTEXT_RATIO
    if new_w < _MIN_CONTEXT_DIM or new_h < _MIN_CONTEXT_DIM:
        return False

    # Lanczos3 is the canonical "best quality downsample" choice — it
    # has the sharpest passband + cleanest stopband of any commonly
    # supported reconstruction filter. pyvips' resize() uses a
    # two-step shrink-then-resample path that gives effectively-
    # Lanczos quality at full pyvips throughput (~200 ms for a 19000-
    # wide source). The result is smooth at every scale the
    # tone-mapping operator samples.
    scale = new_w / w
    context = image.resize(scale, kernel="lanczos3")

    # pyvips returns its internal `band-interleaved float` buffer; we
    # only need the first band for our luminance-only context map.
    raw_f32_bytes = context.write_to_memory()
    flat = np.frombuffer(raw_f32_bytes, dtype=np.float32)
    bands = max(1, len(flat) // (context.width * context.height))
    flat = flat.reshape(context.height, context.width, bands)
    if bands > 1:
        # Luminance reduction: a Rec. 709 gray for color sources, or
        # band-0 for already-mono. Both produce the right "local
        # average brightness" signal the tone-mapping operator wants.
        # Rec. 709 weights pick up perceptual luminance which is what
        # the eye actually adapts to.
        gray = (
            0.2126 * flat[..., 0]
            + 0.7152 * flat[..., 1]
            + 0.0722 * flat[..., 2]
        )
    else:
        gray = flat[..., 0]

    f16 = gray.astype(np.float16)
    payload = (
        _CONTEXT_MAGIC
        + struct.pack("<II", context.width, context.height)
        + f16.tobytes()
    )

    # Brotli quality 11 is the slowest / best-ratio setting. Justified
    # here because the context map is computed exactly ONCE per
    # master (at producer time) and the file is fetched once per
    # image open by the browser — every kilobyte saved at build time
    # pays back across every viewing session afterward.
    compressed = brotli.compress(payload, quality=11)
    tmp = dest_path.with_suffix(dest_path.suffix + ".tmp")
    tmp.write_bytes(compressed)
    os.replace(tmp, dest_path)
    return True


def pyramid_path_for(primary: Path) -> Path:
    """Convention: the display tile pyramid for a deliverable lives at
    the sibling `<stem>_tiles.tiff`.

    The pyramid is an internal display artifact, NOT the user's file.
    Keeping it in its own namespace means a deliverable can itself be a
    `.tiff` (a first-class output format) without colliding with its
    pyramid — the whole reason color images (baked RGB `.tiff`s) and
    mono masters (`.fits`/`.xisf`) are treated uniformly. One function
    owns this name so the builder and the tile-serving resolver can't
    drift."""
    return Path(primary).with_name(Path(primary).stem + "_tiles.tiff")


def needs_display_pyramid(source: Path) -> bool:
    """Whether `source` gets a separate `_tiles.tiff` display pyramid.

    Linear/float deliverables (`.fits`/`.xisf`, and float `.tiff`s like
    the baked-stretch color images) DO — the viewer tone-maps / zooms
    them from a multi-resolution pyramid. Already-display-ready uint8
    `.tiff`s (the autostretched Subframe previews) do NOT: they are their
    own display artifact and serve directly (the tile route's `uchar`
    branch rescales /255), so building a float pyramid from them would
    blow them out. Non-tiff sources are always linear float."""
    source = Path(source)
    if source.suffix.lower() not in (".tif", ".tiff"):
        return True
    try:
        from file_io.tiff_file_io import read_tiff_image_geometry
        return np.issubdtype(read_tiff_image_geometry(source).dtype, np.floating)
    except Exception:
        return True  # unreadable geometry → default to building (safe)


def context_map_path_for(tile_pyramid_path: Path) -> Path:
    """Convention: <pyramid>_tiles.tiff has its context map at
    <pyramid>_tiles.context (brotli-precompressed). One sibling file per
    master; consumers know to fetch the context map alongside the
    pyramid metadata."""
    return tile_pyramid_path.with_suffix(".context")


# ---------------------------------------------------------------------------
# Decorator that wires both artifacts into producer outputs.
# ---------------------------------------------------------------------------

def make_create_tile_pyramid_decorator(
    cache: CacheProtocol,
    make_tile_pyramid: Callable[[Path, Path], None] = default_make_tile_pyramid,
    make_context_map: Callable[[Path, Path], bool] = default_make_context_map,
):
    def create_tile_pyramid(func: Callable = None, *, tile_filename: Callable = None):
        """Decorator that generates a tile pyramid + context map after
        the wrapped function produces an image.

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
            return _wrap(func, tile_filename=None)
        return lambda f: _wrap(f, tile_filename=tile_filename)

    def _wrap(func: Callable, tile_filename: Callable = None):
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
            if not os.path.exists(filename):
                return result

            source = Path(filename)

            if tile_filename is not None:
                tile_pyramid_path = Path(tile_filename(*args, **kwargs))
            elif not needs_display_pyramid(source):
                # Already-display-ready uint8 tiff (Subframe preview): it
                # IS its own display artifact and serves directly. No
                # separate pyramid — the resolver falls back to the master.
                tile_pyramid_path = source
            else:
                # Linear/float deliverable: the pyramid is a separate
                # `_tiles.tiff` sibling, even when the deliverable is itself
                # a `.tiff` — the display artifact never aliases the file.
                tile_pyramid_path = pyramid_path_for(source)
            if tile_pyramid_path != source:
                tile_pyramid_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    make_tile_pyramid(source, tile_pyramid_path)
                except Exception as e:
                    print(f"### Failed to make tile pyramid for {source}: {e}")
                    return result

            cache[("tile_pyramid", str(source))] = str(tile_pyramid_path)

            # Context map: best-effort, doesn't block the pyramid path.
            # Source for the context map is whichever artifact has the
            # linear pixel data — usually the pyramid TIFF, but for
            # producers that write `.tiff` directly (subframe-mode
            # mosaics) it's the producer's own output.
            context_source = tile_pyramid_path
            context_path = context_map_path_for(tile_pyramid_path)
            try:
                wrote = make_context_map(context_source, context_path)
                if wrote:
                    cache[("context_map", str(source))] = str(context_path)
            except Exception as e:
                # Context map is an optional optimization; falling back
                # to source-tile sampling in the shader is correct
                # behavior when it's absent.
                print(f"### Failed to make context map for {source}: {e}")

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


# ---------------------------------------------------------------------------
# Legacy thumbnail path. Kept because /api/thumbnail callers exist on the
# frontend (the small image previews next to dropdowns and list items).
# Reads the lowest pyramid level still >= _THUMBNAIL_MIN_PX so that
# thumbnail rendering doesn't touch the full-resolution pixels.
# ---------------------------------------------------------------------------

_THUMBNAIL_MIN_PX = 1200


def pyramid_thumbnail(tile_pyramid_path, min_px: int = _THUMBNAIL_MIN_PX) -> bytes:
    """PNG bytes of a thumbnail of the tile pyramid.

    Three dispatch paths:

    * **Float pyramid (the new linear-float case):** read the smallest
      pyramid level whose dim is still >= `min_px`, autostretch the
      result to uint8, encode PNG.

    * **uint8 pyramid (legacy autostretched files):** same level
      selection, but the data's already in display space, so we just
      pass it through. (Path kept for backward compat with sidecars
      that survive cache wipes referring to pre-rewrite outputs.)

    * **Single-resolution TIFF (no pyramid levels):** read the full
      file, autostretch if it's float, resize via cv2 to fit `min_px`.
    """
    from previews.operations import autostretch
    import cv2

    path = os.fspath(tile_pyramid_path)
    base = pyvips.Image.tiffload(path, page=0)

    try:
        n_pages = base.get("n-pages")
    except pyvips.Error:
        n_pages = 1

    if n_pages > 1:
        # Pick the lowest-resolution stored level still >= min_px.
        level, dim = 0, min(base.width, base.height)
        while level + 1 < n_pages and dim // 2 >= min_px:
            level += 1
            dim //= 2
        image = pyvips.Image.tiffload(path, page=level)
    else:
        # Single-page TIFF. For float sources we MUST autostretch
        # before any uint8 conversion (otherwise the >255 pixels clip
        # to white, producing the binary-mask failure). Read the full
        # image, autostretch, then cv2-resize to fit.
        if base.format in ("float", "double"):
            arr = read_image(Path(path))
            arr_u8 = (autostretch(arr) * 255.0).astype(np.uint8)
            h, w = arr_u8.shape[:2]
            short = min(h, w)
            if short > min_px:
                scale = min_px / short
                arr_u8 = cv2.resize(
                    arr_u8, (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            ok, buf = cv2.imencode(".png", arr_u8)
            if not ok:
                raise RuntimeError(f"cv2 failed to encode PNG for {path}")
            return buf.tobytes()
        # uint8 single-page TIFF — pyvips.Image.thumbnail does the
        # right thing without losing dynamic range.
        image = pyvips.Image.thumbnail(path, min_px)

    raw = image.write_to_memory()
    bands = image.bands
    vips_to_numpy = {
        "uchar": np.uint8, "char": np.int8,
        "ushort": np.uint16, "short": np.int16,
        "uint": np.uint32, "int": np.int32,
        "float": np.float32, "double": np.float64,
    }
    dtype = vips_to_numpy.get(image.format, np.float32)
    shape = (image.height, image.width, bands) if bands > 1 else (image.height, image.width)
    arr = np.ndarray(buffer=raw, dtype=dtype, shape=shape)

    if dtype == np.uint8:
        arr_u8 = arr
    else:
        arr_u8 = (autostretch(arr) * 255.0).astype(np.uint8)

    ok, buf = cv2.imencode(".png", arr_u8)
    if not ok:
        raise RuntimeError(f"cv2 failed to encode PNG for {path}")
    return buf.tobytes()
