"""HiPS trees: the tile layout against real CDS tiles (recorded), and a tiny
folder tree loaded as a map. Pure numpy + astropy; no app, no Ray, no
network."""
import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from allsky import maps
from allsky.healpix_faces import nested_index, vec_from_lonlat, xyf_from_vec
from allsky.hips import raster_from_tile, read_properties, sample_npix, tile_path
from stacks.post_processing.background_pedestal import DEFAULT_PEDESTAL

TESTDATA = Path(__file__).parent / "testdata"


def test_raster_from_tile_is_a_quarter_turn():
    t = np.arange(16).reshape(4, 4)
    r = raster_from_tile(t)
    for x in range(4):
        for y in range(4):
            assert r[y, x] == t[3 - x, y]
    assert np.array_equal(raster_from_tile(np.rot90(r, 1)), r)     # rot90(·, 1) is the inverse


def test_tile_layout_matches_the_stars_seen_on_cds_tiles():
    """Ten bright stars located by argmax in CDS's unWISE W1 tiles: our
    forward projection must put each in the tile CDS filed it under, at
    FITS column = raster y and row = W−1−raster x (within the saturated
    cores' few-pixel slop)."""
    fx = json.loads((TESTDATA / "hips_tile_layout.json").read_text())
    w = fx["tile_width"]
    for s in fx["stars"]:
        v = np.atleast_2d(vec_from_lonlat(np.array([s["ra"]]), np.array([s["dec"]])))
        face, x, y = (float(a[0]) for a in xyf_from_vec(v))
        n = w << s["order"]
        tx, ty = int(x * n // w), int(y * n // w)
        assert int(nested_index(np.array([int(face)]), np.array([tx]), np.array([ty]),
                                1 << s["order"])[0]) == s["npix"], s["star"]
        lx, ly = x * n - w * tx, y * n - w * ty
        assert abs(s["col"] - ly) < 12 and abs(s["row"] - (w - 1 - lx)) < 12, s["star"]


def test_tile_paths_and_properties():
    assert tile_path(8, 400123) == "Norder8/Dir400000/Npix400123.fits"
    assert tile_path(0, 7, "png") == "Norder0/Dir0/Npix7.png"
    p = read_properties("hips_order = 8\n# a comment\nhips_frame=equatorial\nnoise\n")
    assert p == {"hips_order": "8", "hips_frame": "equatorial"}
    assert sample_npix(0, 24) == list(range(12))
    s = sample_npix(8, 24)
    assert len(s) == 24 and s[0] == 0 and s[-1] == 12 * 4 ** 8 - 1


def test_a_folder_tree_loads_as_a_map(tmp_path, monkeypatch):
    root = tmp_path / "tree"
    (root / "Norder0" / "Dir0").mkdir(parents=True)
    (root / "properties").write_text(
        "hips_order = 0\nhips_tile_width = 4\nhips_frame = equatorial\nhips_tile_format = jpeg fits\n")
    for n in range(12):
        raster = np.full((4, 4), 2.0, np.float32)
        raster[1, 3] = 50.0 + n                       # raster cell x=3, y=1
        raster[2, 2] = np.nan                         # a hole
        fits.PrimaryHDU(np.rot90(raster, 1)).writeto(root / "Norder0" / "Dir0" / f"Npix{n}.fits")
    maps_dir = tmp_path / "maps"
    maps_dir.mkdir()
    (maps_dir / "tiny.hips").write_text(f"{root}\n")
    monkeypatch.setattr(maps, "MAPS_DIR", maps_dir)
    maps._cache.clear()
    m = maps.load_map("tiny")
    assert (m.frame, m.tile_width, m.max_order, m.nside, m.tiles) == ("icrs", 4, 0, 4, "")
    t = m.tile(0, 5)
    assert t[1, 3] == 55.0 and t[0, 0] == 2.0 and t[2, 2] == 0.0
    assert m.stats["skyMedian"] == pytest.approx(DEFAULT_PEDESTAL)
    with pytest.raises(IndexError):
        m.tile(1, 0)
    assert maps.available_maps() == ["tiny"]
    # A tree rebuilt under the same name moves the map's version (the viewer
    # caches tiles as immutable under it) and is measured again.
    import os
    os.utime(root / "properties", (m.mtime + 100, m.mtime + 100))
    again = maps.load_map("tiny")
    assert again is not m and again.mtime == pytest.approx(m.mtime + 100)
