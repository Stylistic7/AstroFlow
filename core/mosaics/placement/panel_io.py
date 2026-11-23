"""On-disk format for placed mosaic panels — "bbox-on-disk".

`place_image` projects a source master into a small footprint bounding box of the
mosaic canvas. Rather than persist a full-canvas TIFF that's almost all zeros — and
force every merger to read/decompress the whole ~9 GB canvas up to 3x — we store
just the bbox pixels, plus the bbox's (x, y) offset in the canvas and the canvas
size, in the TIFF header. The OSS/Subframe mergers read the small array + offset
directly. Panel identity is the call-hash CONTHASH stamp, not a
pixel hash, so this format change does not cascade-invalidate downstream by content.
"""
from pathlib import Path

import numpy as np

from file_io.tiff_file_io import (
    write_tiff_image, read_tiff_image, read_tiff_header, FAST_COMPRESSION,
)

# Header keys recording the bbox's placement in the full mosaic canvas. Plain
# ints round-trip through the JSON description + normalize_header untouched.
_BBOX_X = "BBOX_X"
_BBOX_Y = "BBOX_Y"
_CANVAS_W = "CANVAS_W"
_CANVAS_H = "CANVAS_H"


def write_placed_panel(
    path: Path,
    bbox_image: np.ndarray,
    offset_xy: tuple[int, int],
    canvas_size: tuple[int, int],
    metadata: dict | None = None,
) -> None:
    """Write a placed panel as its bbox pixels + placement metadata, zstd. Read
    by the pyvips OSS/Subframe mergers."""
    x0, y0 = offset_xy
    cw, ch = canvas_size
    meta = dict(metadata or {})
    meta.update({_BBOX_X: int(x0), _BBOX_Y: int(y0), _CANVAS_W: int(cw), _CANVAS_H: int(ch)})
    write_tiff_image(Path(path), bbox_image, meta, compression=FAST_COMPRESSION)


def read_placed_panel(
    path: Path,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Return `(bbox_array, (x0, y0), (canvas_w, canvas_h))` for a placed panel."""
    path = Path(path)
    hdr = read_tiff_header(path)
    arr = read_tiff_image(path)
    return arr, (int(hdr[_BBOX_X]), int(hdr[_BBOX_Y])), (int(hdr[_CANVAS_W]), int(hdr[_CANVAS_H]))


