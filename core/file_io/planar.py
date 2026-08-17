"""On-demand reconstruction of a planar (stripped) TIFF from a tiled pyramid.

WHY THIS EXISTS (2026-08-17). A pyramidal `_tiles.tiff` already contains the
full-resolution payload as page 0 — verified bit-identical to the separately
stored master, metadata included — so keeping both wastes ~267 GB across the
TIFF-payload ops. The masters for the terminal scene formats have been deleted
on that basis.

But **PixInsight cannot read a tiled TIFF.** Measured against PixInsight 1.9.4
by opening four controlled variants of one image:

    BigTIFF + stripped + 1 page   -> opens, 3226x2768
    classic + tiled    + 1 page   -> ZERO windows
    classic + tiled    + pyramid  -> ZERO windows
    BigTIFF + tiled    + pyramid  -> ZERO windows

Tiling is the blocker, not BigTIFF and not multi-page — so a reconstruction has
no 4 GB ceiling and must simply be STRIPPED. Worse, PixInsight fails SILENTLY:
`ImageWindow.open()` returns an empty array rather than raising, so handing it a
pyramid looks like a no-op rather than an error. Hence every "open externally"
path must route through here.

The reconstruction is exact: page 0 is copied unmodified and the
`image-description` (where AstroFlow stamps PIPELINE / COLORSPC / FILTER /
CONTHASH / STBLHASH) is carried across, so the file PixInsight opens is
pixel-for-pixel and metadata-for-metadata the master that used to sit on disk.
"""
from __future__ import annotations

import os
from pathlib import Path

_CACHE_DIRNAME = "astroflow_planar"

# Reconstruction of a gigapixel mosaic is minutes of decompress + recompress
# (measured: 103 s for a 1.06 GB / 8813x15062x3 float32 image), so results are
# cached and reused. Keyed on the source's (path, mtime, size): a rebuilt tap
# moves its mtime, so a stale reconstruction can never be served.
def _cache_dir() -> Path:
    import tempfile
    d = Path(tempfile.gettempdir()) / _CACHE_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_tiled_tiff(path: "str | Path") -> bool:
    """Whether `path` is a TIFF whose first page is TILED (hence unreadable by
    PixInsight). False for non-TIFFs — XISF and FITS go to PixInsight as-is —
    and False when the tags can't be read, so an unreadable file is passed
    through unchanged rather than pointlessly rewritten."""
    p = Path(path)
    if p.suffix.lower() not in (".tif", ".tiff"):
        return False
    try:
        import tifffile
        with tifffile.TiffFile(os.fspath(p)) as tf:
            return bool(tf.pages[0].is_tiled)
    except Exception:
        return False


def planar_path_for(source: "str | Path") -> Path:
    """The cache path a reconstruction of `source` would land at."""
    src = Path(source)
    try:
        st = src.stat()
        stamp = f"{int(st.st_mtime)}_{st.st_size}"
    except OSError:
        stamp = "nostat"
    return _cache_dir() / f"{src.stem}.{stamp}.planar.tiff"


def reconstruct_planar_tiff(source: "str | Path") -> Path:
    """A PixInsight-readable planar TIFF for `source`, reconstructing if needed.

    Returns `source` UNCHANGED when it is not a tiled TIFF (an XISF master, a
    FITS, an already-stripped TIFF) — the common case costs nothing. For a tiled
    pyramid, page 0 is written out stripped, cached on (mtime, size), and that
    path returned.

    The write goes to a temp file and is renamed into place, so a concurrent
    reader never sees a partial reconstruction and two racing callers converge
    on one valid file.
    """
    src = Path(source)
    if not is_tiled_tiff(src):
        return src

    dest = planar_path_for(src)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    import pyvips
    img = pyvips.Image.tiffload(os.fspath(src), page=0, unlimited=True)
    desc = (img.get("image-description")
            if "image-description" in img.get_fields() else None)
    out = img.copy()
    if desc:
        out.set_type(pyvips.GValue.gstr_type, "image-description", desc)

    staging = dest.with_suffix(f".{os.getpid()}.staging")
    try:
        # tile=False is the whole point: PixInsight reads stripped TIFF only.
        # bigtiff=True is safe (measured: BigTIFF opens fine) and necessary
        # since a reconstructed gigapixel mosaic exceeds the 4 GB classic cap.
        out.tiffsave(os.fspath(staging), compression="deflate",
                     predictor="none", bigtiff=True, tile=False)
        os.replace(staging, dest)
    finally:
        if staging.exists():
            try:
                staging.unlink()
            except OSError:
                pass
    return dest
