"""On-disk format for placed mosaic panels — "bbox-on-disk".

`place_image` projects a source master into a small footprint bounding box of the
mosaic canvas. Rather than persist a full-canvas TIFF that's almost all zeros — and
force every merger to read/decompress the whole ~9 GB canvas up to 3x — we store
just the bbox pixels, plus the bbox's (x, y) offset in the canvas and the canvas
size, in the TIFF header. The OSS/Subframe mergers read the small array + offset
directly; the PixInsight PhotometricMosaic path (which needs full-canvas panels
sharing one coordinate frame, and can only read deflate) re-hydrates on demand via
`rehydrate_placed_panel`. Panel identity is the call-hash CONTHASH stamp, not a
pixel hash, so this format change does not cascade-invalidate downstream by content.
"""
from pathlib import Path

import numpy as np

from file_io.tiff_file_io import (
    write_tiff_image, write_tiff_embedded, read_tiff_image, read_tiff_header,
    FAST_COMPRESSION, COMPATIBLE_COMPRESSION,
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
    only by the pyvips OSS/Subframe mergers; the PI path re-hydrates first."""
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


def rehydrate_placed_panel(src_path: Path, dst_path: Path) -> None:
    """Reconstruct the full-canvas deflate TIFF a PixInsight consumer needs
    (PhotometricMosaic infers placement from each image's canvas rect, and PI
    can't read zstd).

    Two input shapes reach here, because `merge_images` chains pairwise:
      * a bbox-on-disk placed panel (TIFF with BBOX_*/CANVAS_* headers) — embed
        the bbox on a black canvas, exactly what place_image wrote pre-bbox;
      * an already-full-canvas master from a PRIOR `merge_images` (XISF, no bbox
        headers) — a straight transcode to deflate TIFF (it's already in the
        mosaic frame, so no embedding).
    Reads are suffix-dispatched (`file_io.generic`) so the XISF case doesn't hit
    the TIFF-only pyvips loader (which has no `xisfload`)."""
    from file_io.generic import read_header, read_image
    src_path = Path(src_path)
    hdr = read_header(src_path)
    arr = read_image(src_path)
    if _BBOX_X in hdr:
        write_tiff_embedded(
            Path(dst_path), arr,
            offset_xy=(int(hdr[_BBOX_X]), int(hdr[_BBOX_Y])),
            canvas_size=(int(hdr[_CANVAS_W]), int(hdr[_CANVAS_H])),
            compression=COMPATIBLE_COMPRESSION)
    else:
        write_tiff_image(Path(dst_path), arr, dict(hdr),
                         compression=COMPATIBLE_COMPRESSION)
