"""The packed tile planes: tiles written by the shard come back, a tile never
written is None and costs no file, the fill value removes one, coverage and
per-tile scalars follow, the smallest orders fit in one shard, and the
properties move the version. Pure numpy + zarr; no app, no Ray."""
import glob
import os
import time

import numpy as np
import pytest

from allsky import tiles


def _files(root):
    return sorted(os.path.relpath(p, root) for p in glob.glob(f"{root}/**", recursive=True)
                  if os.path.isfile(p) and not p.endswith("zarr.json"))


def test_tiles_round_trip_by_shard(tmp_path):
    planes = tiles.create(tmp_path, {"mean": (np.float32, 8), "count": (np.uint16, 4)}, order=2,
                          scalars={"mean": ("psffwhm",)})
    mean, count = planes["mean"], planes["count"]
    a = np.full((8, 8), 1.5, np.float32)
    a[2, 3] = np.nan                                    # a hole inside a tile
    b = np.full((8, 8), 2.5, np.float32)
    s = tiles.shard_of(70)
    assert s == tiles.shard_of(71) == 1 and tiles.shard_range(2, 1) == (64, 128)
    mean.write_block(2, 1, {70: a, 71: b}, {"psffwhm": {70: 3.25, 71: 4.0}})
    count.write_block(2, 1, {70: np.full((4, 4), 2, np.uint16), 71: np.ones((4, 4), np.uint16)})
    assert _files(tmp_path / "mean") == ["2/c/1/0/0", "coverage_2/c/1", "psffwhm_2/c/1"]   # one shard, nothing else
    r = tiles.Plane(tmp_path / "mean")
    got = r.tile(2, 70)
    assert got.dtype == np.float32 and got[0, 0] == 1.5 and np.isnan(got[2, 3])
    assert r.tile(2, 71)[0, 0] == 2.5 and r.tile(2, 72) is None and r.tile(2, 0) is None
    assert r.existing(2) == [70, 71] and r.existing(1) == []
    assert r.scalar("psffwhm", 2, 70) == pytest.approx(3.25)
    c = tiles.Plane(tmp_path / "count").tile(2, 70)
    assert c.dtype == np.uint16 and c[0, 0] == 2
    with pytest.raises(IndexError):
        r.tile(2, 12 * 16)
    with pytest.raises(ValueError):
        mean.write_block(2, 0, {70: a})                   # 70 is shard 1's


def test_the_fill_value_removes_a_tile_and_the_rest_of_the_shard_stays(tmp_path):
    plane = tiles.create(tmp_path, {"mean": (np.float32, 4)}, order=3)["mean"]
    tiles_in = {n: np.full((4, 4), float(n), np.float32) for n in (128, 129, 130)}
    plane.write_block(3, 2, tiles_in)
    plane.write_block(3, 2, {129: None})
    r = tiles.Plane(tmp_path / "mean")
    assert r.existing(3) == [128, 130] and r.tile(3, 129) is None
    assert r.tile(3, 128)[0, 0] == 128.0 and r.tile(3, 130)[0, 0] == 130.0
    plane.write_block(3, 2, {128: None, 130: None})
    assert tiles.Plane(tmp_path / "mean").existing(3) == []
    assert _files(tmp_path / "mean") == []               # an empty shard is not stored


def test_the_coarse_orders_fit_in_one_shard(tmp_path):
    plane = tiles.create(tmp_path, {"mean": (np.float32, 4)}, order=1)["mean"]
    assert tiles.shard_range(0, 0) == (0, 12) and tiles.shard_range(1, 0) == (0, 48)
    plane.write_block(0, 0, {n: np.full((4, 4), float(n), np.float32) for n in range(12)})
    r = tiles.Plane(tmp_path / "mean")
    assert r.existing(0) == list(range(12)) and r.tile(0, 11)[3, 3] == 11.0
    assert _files(tmp_path / "mean") == ["0/c/0/0/0", "coverage_0/c/0"]


def test_properties_are_the_version_and_come_last(tmp_path):
    plane = tiles.create(tmp_path, {"mean": (np.float32, 4)}, order=0)["mean"]
    plane.write_block(0, 0, {3: np.ones((4, 4), np.float32)})
    v0 = plane.version
    time.sleep(0.02)
    plane.write_properties({"hips_order": 0, "hips_tile_width": 4, "hips_frame": "equatorial"})
    r = tiles.Plane(tmp_path / "mean")
    assert r.properties == {"hips_order": "0", "hips_tile_width": "4", "hips_frame": "equatorial"}
    assert r.version > v0 and tiles.is_plane(tmp_path / "mean") and not tiles.is_plane(tmp_path)
    # Opened again for writing, the arrays are found, not recreated.
    again = tiles.create(tmp_path, {"mean": (np.float32, 4)}, order=0)["mean"]
    assert again.tile(0, 3)[0, 0] == 1.0
