"""`viewable_from_master` must never hand the tile route a float master.

2026-08-14: a 19 GB float `Linear RGB` master whose pyramid was still building
was served directly, because the fallback accepted ANY `.tiff`. Two things go
wrong when it does:

  * the master is ONE page, so the tile route's `page = maxZoom - z` goes
    negative and pyvips — which warns `value "-8" out of range for property
    'page'` rather than raising — silently serves FULL RESOLUTION at every
    zoom level;
  * it is written strip-based (Rows/Strip: 128), so random tile access decodes
    ~74 MB strips (the 822 s `channelStats` requests).

Already-display-ready uint8 tiffs (Subframe previews) ARE their own display
artifact and must still serve directly — the tile route's `uchar` branch
rescales /255. That distinction is `needs_display_pyramid`.
"""
import numpy as np
import pytest

from catalog.queries import viewable_from_master
from previews.create_tile_pyramid import pyramid_path_for


def _tiff(path, dtype, w=32, h=32):
    import pyvips
    if dtype == "float":
        a = (np.random.default_rng(0).random((h, w, 3)) * 0.1).astype("float32")
        fmt = "float"
    else:
        a = (np.random.default_rng(0).random((h, w, 3)) * 255).astype("uint8")
        fmt = "uchar"
    pyvips.Image.new_from_memory(a.tobytes(), w, h, 3, fmt).tiffsave(str(path))
    return path


class TestViewableGating:
    def test_float_master_without_a_pyramid_is_not_viewable(self, tmp_path):
        m = _tiff(tmp_path / "m.tiff", "float")
        assert viewable_from_master(m) is None, \
            "a float master must never be served as tiles"

    def test_float_master_with_a_pyramid_serves_the_pyramid(self, tmp_path):
        m = _tiff(tmp_path / "m.tiff", "float")
        pyr = _tiff(pyramid_path_for(m), "float")
        assert viewable_from_master(m) == pyr

    def test_uint8_master_without_a_pyramid_still_serves_directly(self, tmp_path):
        """Subframe previews are their own display artifact — unchanged."""
        m = _tiff(tmp_path / "u.tiff", "uint8")
        assert viewable_from_master(m) == m

    def test_non_tiff_master_without_a_pyramid_is_not_viewable(self, tmp_path):
        m = tmp_path / "m.fits"
        m.write_bytes(b"not really a fits")
        assert viewable_from_master(m) is None

    def test_none_and_missing_are_not_viewable(self, tmp_path):
        assert viewable_from_master(None) is None
        assert viewable_from_master(tmp_path / "absent.tiff") is None
