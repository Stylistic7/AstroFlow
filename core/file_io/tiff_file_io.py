"""TIFF read/write, backed entirely by libvips (pyvips).

pyvips replaced tifffile here because it streams tile-by-tile (memory-bounded on
gigapixel masters), is already the TIFF path everywhere else (deliverables, tile
pyramids, tile-serving), and doesn't share tifffile's BigTIFF footgun — classic
TIFF caps offsets at 4 GB (32-bit) and tifffile's auto-detection under-triggered
with compression, overflowing on 9 GB channels. We force BigTIFF for large
images ourselves.

Metadata rides on the TIFF ImageDescription tag (270): libvips calls it
`image-description`, tifffile called it `description` — same tag, so files
round-trip both ways and old tifffile-written files still read.
"""
import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pyvips
import tifffile

from file_io.normalize_header import normalize_header
from file_io.types import ImageGeometry

# numpy dtype <-> libvips band format. Covers every dtype AstroFlow writes to
# TIFF (float32 masters/color, uint8/uint16 previews).
_NP_TO_VIPS = {
    "uint8": "uchar", "int8": "char",
    "uint16": "ushort", "int16": "short",
    "uint32": "uint", "int32": "int",
    "float32": "float", "float64": "double",
    "complex64": "complex", "complex128": "dpcomplex",
}
_VIPS_TO_NP = {v: k for k, v in _NP_TO_VIPS.items()}
_VIPS_FMT_BYTES = {
    "uchar": 1, "char": 1, "ushort": 2, "short": 2, "uint": 4, "int": 4,
    "float": 4, "double": 8, "complex": 8, "dpcomplex": 16,
}

# libvips name for the ImageDescription tag (== tifffile's `description`).
_DESC = "image-description"
# Classic TIFF offsets are 32-bit (4 GB); force BigTIFF well before that. libvips
# does NOT auto-enable BigTIFF, so we decide from the pixel budget.
_BIGTIFF_THRESHOLD = 2 * 1024**3
# DEFAULT TIFF codec: zstd, multi-core — ~18x faster than deflate's
# single-threaded zlib on dense gigapixel data (deflate pegs one core for ~12 min
# on a 27 GB color TIFF; zstd ~40 s; a 1.3 Gpx SHO join went 33 min -> 3.4 min)
# and smaller on our float data. libvips reads it transparently; external tools
# (PixInsight, rc-astro, astrometry.net) cannot, so every hand-off to one goes
# through `file_io.planar` (`needs_planar` / `write_planar`): a stripped
# uncompressed copy written at the boundary (Stylistic7 2026-10-11: AstroFlow runs
# fast, the PixInsight export may take a minute).
FAST_COMPRESSION = "zstd"
# deflate: what an external consumer that needs a COMPRESSED full file gets
# (the PhotometricMosaic re-hydration in mosaics/placement/panel_io.py).
COMPATIBLE_COMPRESSION = "deflate"


def _open(filename: Path) -> pyvips.Image:
    # revalidate=True bypasses libvips's operation cache so a file that was
    # overwritten in place is never read stale. Without it, a read that
    # populated the cache BEFORE an in-place rewrite (e.g. update_tiff_header
    # opens the file to merge metadata, then overwrites it with the CONTHASH
    # stamp) returns the OLD header on the next same-process read — so produced
    # images silently lose their content-hash stamp and get_content_hash falls
    # back to pixel-hashing, giving them a non-deterministic identity that
    # never cache-hits (the "re-derives all night" churn).
    #
    # unlimited=True disables libvips's anti-decompression-bomb guard, a fixed
    # ~50 MB cap on the cumulated libtiff scratch allocation for a *single*
    # strip/tile decode. Our masters are wide (a 48206px-wide float32×3 image at
    # 128 rows/strip = ~74 MB per strip), so a legitimate read of one strip
    # trips the cap and aborts ("No space for data buffer"). These are our own
    # trusted pipeline outputs, so the DoS guard buys us nothing.
    return pyvips.Image.new_from_file(str(filename), revalidate=True, unlimited=True)


def _get_description(img: pyvips.Image) -> str:
    try:
        return img.get(_DESC) or "{}"
    except pyvips.Error:
        return "{}"


def _vips_from_numpy(arr: np.ndarray) -> pyvips.Image:
    """Wrap a numpy array as a vips image. NOTE: `new_from_memory` references the
    buffer, so the caller MUST keep `arr` alive until the save completes."""
    if arr.ndim == 2:
        height, width, bands = arr.shape[0], arr.shape[1], 1
    elif arr.ndim == 3:
        height, width, bands = arr.shape
    else:
        raise ValueError(f"TIFF write expects a 2D or 3D array, got {arr.shape}")
    fmt = _NP_TO_VIPS.get(str(arr.dtype))
    if fmt is None:
        raise ValueError(f"unsupported dtype for TIFF write: {arr.dtype}")
    return pyvips.Image.new_from_memory(arr.data, width, height, bands, fmt)


def _atomic_tiffsave(
    img: pyvips.Image,
    filename: Path,
    tile: tuple[int, int] | None = None,
    compression: str = FAST_COMPRESSION,
) -> None:
    """Save via a sibling tmp + atomic rename (readers never see a half-written
    file). zstd by default (see FAST_COMPRESSION). BigTIFF forced for large
    images."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    tmp = filename.with_name(f".{filename.name}.{uuid.uuid4().hex}.tmp")
    nbytes = img.width * img.height * img.bands * _VIPS_FMT_BYTES.get(img.format, 4)
    tile_kwargs = (
        {"tile": True, "tile_width": tile[0], "tile_height": tile[1]}
        if tile is not None else {}
    )
    try:
        img.tiffsave(
            str(tmp), compression=compression,
            bigtiff=nbytes >= _BIGTIFF_THRESHOLD, **tile_kwargs,
        )
        os.replace(tmp, filename)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def read_tiff_header(filename: Path) -> dict:
    return normalize_header(json.loads(_get_description(_open(filename))))


def read_tiff_image_geometry(filename: Path) -> ImageGeometry:
    img = _open(filename)
    dtype = np.dtype(_VIPS_TO_NP[img.format])
    shape = (
        (img.height, img.width) if img.bands == 1
        else (img.height, img.width, img.bands)
    )
    return ImageGeometry.from_shape_and_dtype(shape, dtype)


def read_tiff_image(filename: Path) -> np.ndarray:
    arr = _open(filename).numpy()  # (H, W, bands)
    # Single-band TIFFs read back as 2D (H, W), matching the old tifffile
    # behaviour consumers rely on.
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return arr


def write_tiff_image(
    filename: Path,
    image: np.ndarray,
    metadata: dict[str, Any] | None = None,
    tile: tuple[int, int] | None = None,
    compression: str = FAST_COMPRESSION,
    **_legacy: Any,  # absorbs old tifffile-only kwargs (e.g. bigtiff); ignored
) -> None:
    metadata = metadata or {}
    arr = np.ascontiguousarray(image)  # kept alive through the tiffsave below
    img = _vips_from_numpy(arr).copy()
    img.set_type(pyvips.GValue.gstr_type, _DESC, json.dumps(metadata, ensure_ascii=True))
    _atomic_tiffsave(img, Path(filename), tile=tile, compression=compression)
    assert Path(filename).exists()
    del arr  # explicit: buffer no longer referenced after the save


def write_tiff_image_streaming(
    filename: Path,
    image: "pyvips.Image",
    metadata: dict[str, Any] | None = None,
    tile: tuple[int, int] | None = None,
    compression: str = FAST_COMPRESSION,
) -> None:
    """`write_tiff_image` for a LAZY pyvips image: tiffsave pulls the
    pipeline tile-by-tile, so a gigapixel result never materializes in
    RAM (the `write_tiff_embedded` idiom generalized to any lazy
    expression). The caller owns keeping any borrowed numpy buffers in
    the pipeline alive until this returns."""
    img = image.copy()
    img.set_type(pyvips.GValue.gstr_type, _DESC,
                 json.dumps(metadata or {}, ensure_ascii=True))
    _atomic_tiffsave(img, Path(filename), tile=tile, compression=compression)
    assert Path(filename).exists()


def write_tiff_embedded(
    filename: Path,
    image: np.ndarray,
    offset_xy: tuple[int, int],
    canvas_size: tuple[int, int],
    metadata: dict[str, Any] | None = None,
    compression: str = FAST_COMPRESSION,
) -> None:
    """Write a full `canvas_size` (W, H) TIFF holding `image` at `offset_xy`
    (x, y) with the rest black — WITHOUT ever materializing the full canvas.

    libvips `embed` builds a lazy canvas whose out-of-image region is
    synthesized as zeros tile-by-tile on demand, so `tiffsave` streams the
    whole (possibly gigapixel) output while only `image` sits in RAM. This is
    what lets `place_image` keep its RSS at panel-bbox size instead of
    full-mosaic-canvas size: a placed panel fills only a small bbox of the
    canvas, the rest is `place_image`-zero anyway (the OSS merge relies on
    exactly that), so the bytes on disk are identical to writing a
    fully-materialized zero-padded canvas — just without the ~9 GB array.
    """
    metadata = metadata or {}
    arr = np.ascontiguousarray(image)  # kept alive through the tiffsave below
    x0, y0 = offset_xy
    canvas_w, canvas_h = canvas_size
    img = _vips_from_numpy(arr).embed(
        x0, y0, canvas_w, canvas_h, extend="black"
    ).copy()
    img.set_type(pyvips.GValue.gstr_type, _DESC, json.dumps(metadata, ensure_ascii=True))
    _atomic_tiffsave(img, Path(filename), compression=compression)
    assert Path(filename).exists()
    del arr  # explicit: buffer no longer referenced after the save


def update_tiff_header(filename: Path, metadata_in: dict[str, Any]) -> None:
    """Merge `metadata_in` into the TIFF's description, editing the tag IN
    PLACE: tifffile writes the new string (appended at EOF when longer) and
    then repoints the tag, so the pixel data, codec and tile layout are left
    exactly as written. Re-encoding instead cost a single-threaded deflate of
    the whole file per stamp (15 min on a 9.4 GB SHO join) and silently
    turned tiled outputs into strips. A file with no description tag (only
    the untagged exports/previews) has no tag to edit and is re-encoded."""
    img = _open(filename)
    metadata = normalize_header(json.loads(_get_description(img)))
    metadata |= metadata_in
    description = json.dumps(metadata, ensure_ascii=True)
    try:
        tifffile.tiffcomment(filename, description)
    except ValueError:  # no ImageDescription tag
        img = img.copy()
        img.set_type(pyvips.GValue.gstr_type, _DESC, description)
        _atomic_tiffsave(img, Path(filename))
