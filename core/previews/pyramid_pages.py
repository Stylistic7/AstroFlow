"""Zoom-level pages of a servable image, shared by the tile server and the
server-side finish baker: a stored TIFF pyramid page, or an XISF master
shrunk on the fly from its memory map."""
import os
import threading

import pyvips

# (path, page) -> (pyvips.Image, mtime). Pyramid pages are lazy-decode
# handles, so caching them is cheap. Page 0 is always the full-resolution
# page; the metadata endpoint also uses it to compute width/height/levels.
_vips_page_cache: dict = {}
_vips_page_cache_lock = threading.Lock()


def _get_vips_page(path: str, page: int):
    """The image for one zoom level.

    TIFF: the stored pyramid page. XISF: there IS no stored pyramid — the
    master is memory-mapped and shrunk by 2**page on the fly, which measured
    FASTER than reading a pyramid page off disk (0.4-0.9 ms/tile vs 1-6 ms)
    while costing 0 GB and no build time. See `file_io/xisf_mmap`.

    The cache holds the mapping alive alongside the image: `new_from_memory`
    does not take ownership, so dropping the buffer under a live vips image is
    a segfault. Entries are keyed on mtime, so a rebuilt master re-maps.
    """
    current_mtime = os.path.getmtime(path)
    key = (path, page)
    with _vips_page_cache_lock:
        cached = _vips_page_cache.get(key)
        if cached is not None:
            img, cached_mtime, _keep = cached
            if cached_mtime == current_mtime:
                return img
        if str(path).lower().endswith(".xisf"):
            from file_io.xisf_mmap import open_vips
            base, keep = open_vips(path)
            img = base if page == 0 else base.shrink(2 ** page, 2 ** page)
        else:
            img, keep = pyvips.Image.tiffload(
                path, page=page, unlimited=True), None
        _vips_page_cache[key] = (img, current_mtime, keep)
        return img


def _pyramid_levels(path: str) -> int:
    """Number of zoom levels. TIFF reports its stored pages; XISF has none
    stored, so the count is derived from geometry exactly as
    `tiffsave(pyramid=True)` would — the viewer's zoom range must not depend on
    the storage format."""
    base = _get_vips_page(path, 0)
    if str(path).lower().endswith(".xisf"):
        from file_io.xisf_mmap import pyramid_levels
        return pyramid_levels(base.width, base.height)
    try:
        return int(base.get("n-pages"))
    except pyvips.Error:
        return 1
