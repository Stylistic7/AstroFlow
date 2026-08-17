"""Serve XISF masters straight from the file, with no pyramid on disk.

WHY (2026-08-17). Every XISF display tap used to get a sibling `_tiles.tiff`
pyramid: 1,191 files, 273.8 GB, and ~13 s of build time each — 4.3 hours per
full run — to serve displays that received ZERO tile requests across every app
log we have. Measured alternative:

    XISF(path)                 5.1 ms   header only, no pixels
    np.memmap + vips wrap      3.7 ms   RSS +0 MB
    tile from the mapping      0.4-0.9 ms   (FASTER than reading a pyramid)

The pixel block is raw, uncompressed, contiguous float32 at a known offset
(`location="attachment:20480:244417800"`, and 9570*6385*4 = 244417800 exactly),
so `np.memmap` maps it directly and libvips reads through the mapping.

THE POINT IS THE MEMORY BEHAVIOUR, not the speed. A decoded array is anonymous
memory: freeing it does NOT return RSS (measured — three alloc/free cycles all
plateaued at 495 MB), so an in-process LRU with a TTL would cap growth but never
give RAM back. Mapped pages are FILE-BACKED: the OS drops them under pressure,
instantly, without swap and without an allocator high-water mark. That is why
there is no cache, no TTL and no eviction thread in this module — the page cache
already is one, and a better one.

libvips has no XISF loader, so this is the only way in short of a full decode.
"""
from __future__ import annotations

import os
from pathlib import Path

#: Preconditions for the fast path. Verified across 240 sampled masters from
#: stack_files / post_process_stack / post_process_stack_graxpert /
#: merge_target_channel: 100% uncompressed, `attachment` location, default
#: little-endian, single-channel Gray. Anything else falls back to the library's
#: own decode — reading a compressed block as raw floats would yield plausible
#: GARBAGE rather than an error, so these checks are load-bearing.
_SUPPORTED_LOCATION = "attachment"


def _describe(path: Path) -> "dict | None":
    """Geometry + data-block location for `path`, using the xisf package's own
    parser (5 ms, header only — it does not touch pixels). None when the header
    can't be read at all."""
    try:
        from xisf import XISF
        meta = XISF(os.fspath(path)).get_images_metadata()
        return meta[0] if meta else None
    except Exception:
        return None


def can_mmap(path: "str | Path") -> bool:
    """Whether `path` is an XISF whose pixels can be mapped directly."""
    p = Path(path)
    if p.suffix.lower() != ".xisf":
        return False
    m = _describe(p)
    if not m:
        return False
    loc = m.get("location") or ()
    if not loc or loc[0] != _SUPPORTED_LOCATION:
        return False
    if m.get("compression"):
        return False
    geom = m.get("geometry") or ()
    return len(geom) == 3 and geom[2] == 1


def open_vips(path: "str | Path"):
    """`(vips_image, keepalive)` for an XISF master.

    The fast path memory-maps the pixel block; the fallback decodes through the
    xisf package. EITHER WAY the second element must be kept alive for as long
    as the image is used: `Image.new_from_memory` does NOT copy or take
    ownership, so releasing the buffer while vips still holds it is a SEGFAULT,
    not an exception.
    """
    import numpy as np
    import pyvips

    p = Path(path)
    m = _describe(p)
    if m is not None and can_mmap(p):
        w, h, c = m["geometry"]          # XISF order is (W, H, C) — numpy wants (H, W)
        _, offset, _nbytes = m["location"]
        buf = np.memmap(os.fspath(p), dtype=m["dtype"], mode="r",
                        offset=offset, shape=(h, w))
        img = pyvips.Image.new_from_memory(buf, w, h, c, "float")
        return img, buf

    # Fallback: compressed, non-attachment, multi-channel, or unreadable header.
    from xisf import XISF
    arr = XISF(os.fspath(p)).read_image(0)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    h, w, c = arr.shape
    buf = np.ascontiguousarray(arr, dtype="float32")
    img = pyvips.Image.new_from_memory(buf, w, h, c, "float")
    return img, buf


def pyramid_levels(width: int, height: int, tile: int = 256) -> int:
    """How many zoom levels a pyramid over this geometry WOULD have.

    Serving an XISF has no stored pyramid, so the tile route derives the level
    count the same way `tiffsave(pyramid=True)` would: halve until the image
    fits one tile. Keeping this identical to the stored-pyramid case means the
    viewer's zoom range does not change with the storage format.
    """
    levels = 1
    while width > tile or height > tile:
        width = (width + 1) // 2
        height = (height + 1) // 2
        levels += 1
    return levels
