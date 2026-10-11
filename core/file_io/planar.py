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
no 4 GB ceiling and must simply be STRIPPED. The codec is the other blocker:
AstroFlow writes zstd (multi-core), which PixInsight and rc-astro cannot decode,
so a stripped zstd TIFF is rewritten too (`needs_planar`). Worse, PixInsight fails SILENTLY:
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


# The codecs every TIFF reader decodes (PixInsight, rc-astro, astrometry.net).
_PORTABLE_CODECS = {"NONE", "ADOBE_DEFLATE", "DEFLATE", "LZW"}


def needs_planar(path: "str | Path") -> bool:
    """Whether an external tool needs a rewritten copy of `path`: a TIFF that is
    tiled or uses a codec outside `_PORTABLE_CODECS` (AstroFlow's zstd). False
    for non-TIFFs and for unreadable tags, exactly as `is_tiled_tiff`."""
    p = Path(path)
    if p.suffix.lower() not in (".tif", ".tiff"):
        return False
    try:
        import tifffile
        with tifffile.TiffFile(os.fspath(p)) as tf:
            page = tf.pages[0]
            return bool(page.is_tiled) or page.compression.name not in _PORTABLE_CODECS
    except Exception:
        return False


def write_planar(source: "str | Path", dest: "str | Path") -> None:
    """Write page 0 of `source` to `dest` as a stripped, uncompressed BigTIFF,
    carrying the `image-description` across — the copy every external tool
    reads. Atomic: staged beside `dest`, then renamed."""
    import pyvips
    dest = Path(dest)
    img = pyvips.Image.tiffload(os.fspath(source), page=0, unlimited=True)
    desc = (img.get("image-description")
            if "image-description" in img.get_fields() else None)
    out = img.copy()
    if desc:
        out.set_type(pyvips.GValue.gstr_type, "image-description", desc)

    staging = dest.with_suffix(f".{os.getpid()}.staging")
    try:
        # tile=False is the whole point: PixInsight reads stripped TIFF only.
        # BigTIFF only past 2 GB, as the writers do (PixInsight opens BigTIFF;
        # rc-astro and astrometry.net have only ever been fed classic below it).
        # Uncompressed: this copy exists to be READ ONCE by PixInsight, and
        # strip encoding is single-threaded — deflate wrote 30 MB/s (a 7.7 GB
        # channel in six minutes, the 23.5 GB SHO join in half an hour, inside
        # a web request); uncompressed streams at disk speed, 386 MB/s
        # measured, for a file only ~12% larger (float sky compresses poorly).
        from file_io.tiff_file_io import _BIGTIFF_THRESHOLD, _VIPS_FMT_BYTES
        nbytes = out.width * out.height * out.bands * _VIPS_FMT_BYTES.get(out.format, 4)
        out.tiffsave(os.fspath(staging), compression="none", tile=False,
                     bigtiff=nbytes >= _BIGTIFF_THRESHOLD)
        os.replace(staging, dest)
    finally:
        if staging.exists():
            try:
                staging.unlink()
            except OSError:
                pass


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

    Returns `source` UNCHANGED when no rewrite is needed (an XISF master, a
    FITS, a stripped deflate TIFF) — the common case costs nothing. Otherwise
    page 0 is written out stripped and uncompressed (`write_planar`), cached on
    (mtime, size), and that path returned. Two racing callers converge on one
    valid file: the write is staged and renamed.
    """
    src = Path(source)
    if not needs_planar(src):
        return src

    dest = planar_path_for(src)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    write_planar(src, dest)
    return dest
