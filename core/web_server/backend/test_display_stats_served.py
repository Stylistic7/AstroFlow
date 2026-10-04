"""Exact display stats reach the viewer (2026-10-04): a pyramid stores its
full-resolution stats in its sidecar when written; the tile server serves
them while they describe the file's current bytes, ignores them once it
changes, measures an XISF master on the fly, and for a star composite takes
the sky from the exact starless stats."""
import os
import time

import numpy as np
import pyvips
import pytest

from previews.create_tile_pyramid import default_make_tile_pyramid
from previews.display_stats import display_stats, stored_display_stats
from web_server.backend import tiles

PED = 0.005


def _src(tmp_path, name, arr):
    p = tmp_path / f"{name}.src.tiff"
    a = np.ascontiguousarray(arr, np.float32)
    pyvips.Image.new_from_memory(a.data, a.shape[1], a.shape[0], a.shape[2], "float").tiffsave(str(p))
    return p


@pytest.fixture
def scene():
    rng = np.random.default_rng(5)
    a = PED + rng.normal(0, 1e-5, (600, 700, 3))
    a[100:250, 200:400] += 5e-4
    return a.astype(np.float32)


def test_pyramid_stores_exact_stats_that_the_server_serves(tmp_path, scene):
    tiles._stats_cache.clear()
    pyr = tmp_path / "view.tiff"
    default_make_tile_pyramid(_src(tmp_path, "view", scene), pyr)
    stored = stored_display_stats(pyr)
    assert stored == display_stats(pyvips.Image.tiffload(str(pyr), page=0),
                                   coarse=pyvips.Image.tiffload(str(pyr), page=0))
    assert tiles._per_channel_linear_stats(str(pyr)) == stored


def test_stats_measured_before_a_rename_validate_at_the_served_path(tmp_path, scene):
    scratch, final = tmp_path / ".v.staging.pyr.tiff", tmp_path / "v.tiff"
    default_make_tile_pyramid(_src(tmp_path, "v", scene), scratch, served=final)
    assert stored_display_stats(final) is None          # not landed yet
    os.replace(scratch, final)
    assert stored_display_stats(final) is not None


def test_stale_stats_are_ignored(tmp_path, scene):
    pyr = tmp_path / "s.tiff"
    default_make_tile_pyramid(_src(tmp_path, "s", scene), pyr)
    time.sleep(0.01)
    os.utime(pyr)                                        # the file changed
    assert stored_display_stats(pyr) is None


def test_star_composite_takes_the_sky_from_the_exact_stats(tmp_path, scene):
    tiles._stats_cache.clear()
    sl = tmp_path / "sl.tiff"
    default_make_tile_pyramid(_src(tmp_path, "sl", scene), sl)
    st = np.zeros_like(scene)
    st[300:303, 300:303] = 0.4
    stars = tmp_path / "st.tiff"
    default_make_tile_pyramid(_src(tmp_path, "st", st), stars)
    exact = stored_display_stats(sl)
    comp = tiles._per_channel_linear_stats(str(sl), star_paths=[str(stars)])
    for c, e in zip(comp, exact):
        assert (c["skyMedian"], c["skySigma"], c["median"], c["sigma"]) == \
               (e["skyMedian"], e["skySigma"], e["median"], e["sigma"])
        assert c["max"] > 0.3                            # the bright end sees the stars


def test_xisf_master_is_measured_on_the_fly(tmp_path, scene):
    from file_io.xisf_file_io import write_xisf_image
    tiles._stats_cache.clear()
    p = tmp_path / "m.xisf"
    write_xisf_image(p, np.ascontiguousarray(scene[..., 1:2]))
    got = tiles._per_channel_linear_stats(str(p))
    assert len(got) == 1
    v = np.sort(scene[..., 1].ravel())
    assert got[0]["median"] == float(v[int(round(0.5 * (v.size - 1)))])
