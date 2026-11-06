"""The map loader on a small synthetic NESTED map written to a temp folder.
Pure numpy + astropy; no app, no Ray."""
import numpy as np
import pytest
from astropy.io import fits

from allsky import maps
from allsky.healpix_faces import nested_index
from stacks.post_processing.background_pedestal import DEFAULT_PEDESTAL


def _write_map(path, values, nside, ordering="NESTED", coordsys=None, unit="R"):
    col = fits.Column(name="TEMPERATURE", format="E", unit=unit, array=values)
    hdu = fits.BinTableHDU.from_columns([col])
    hdu.header["PIXTYPE"] = "HEALPIX"
    hdu.header["ORDERING"] = ordering
    hdu.header["NSIDE"] = nside
    if coordsys:
        hdu.header["COORDSYS"] = coordsys
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)


@pytest.fixture
def maps_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(maps, "MAPS_DIR", tmp_path)
    monkeypatch.setattr(maps, "TILE_WIDTH", 4)       # a two-level pyramid from an nside-8 map
    maps._cache.clear()
    return tmp_path


def test_map_is_served_on_the_pinned_gauge_as_tiles(maps_dir):
    nside = 8
    rng = np.random.default_rng(1)
    values = rng.lognormal(np.log(3.0), 0.3, 12 * nside * nside).astype(np.float32)
    values[5] = 400.0                                      # one bright cell
    _write_map(maps_dir / "demo.fits", values, nside)
    m = maps.load_map("demo")
    assert (m.nside, m.frame, m.unit, m.tile_width, m.max_order, m.tiles) == (8, "galactic", "R", 4, 1, "")
    # The sky median lands on the stacks' pedestal; every stat is scaled with it.
    assert m.stats["skyMedian"] == pytest.approx(DEFAULT_PEDESTAL)
    assert m.stats["max"] == pytest.approx(400.0 * m.gauge)
    # Tile (order 1, face 3, tx 1, ty 0) holds raster cells x 4..7, y 0..3 of
    # face 3, in file units: position (x=5, y=2) is NESTED cell 3·8² + morton(5, 2).
    npix = int(nested_index(3, 1, 0, 2))
    i = int(nested_index(3, 5, 2, nside))
    assert m.tile(1, npix)[2, 1] == pytest.approx(values[i])
    # Order 0 is the 2×2 mean of order 1.
    fine = np.block([[m.tile(1, int(nested_index(3, tx, ty, 2))) for tx in (0, 1)] for ty in (0, 1)])
    assert m.tile(0, 3) == pytest.approx(fine.reshape(4, 2, 4, 2).mean(axis=(1, 3)))
    for bad in [(2, 0), (0, 12), (1, 48), (-1, 0)]:
        with pytest.raises(IndexError):
            m.tile(*bad)


def test_a_wide_map_is_scaled_so_its_anchor_fits_the_finishes(maps_dir):
    nside = 8
    values = np.full(12 * nside * nside, 1.0, np.float32)
    values[:300] = np.linspace(500.0, 5000.0, 300)           # a bright plane, 9+ stops up
    _write_map(maps_dir / "wide.fits", values, nside)
    m = maps.load_map("wide")
    assert m.stats["high"] == pytest.approx(maps.TOP_CEILING)
    assert m.stats["skyMedian"] < DEFAULT_PEDESTAL


def test_unseen_cells_become_the_no_data_zero(maps_dir):
    nside = 2
    values = np.full(12 * nside * nside, 2.0, np.float32)
    values[7] = -1.6375e30
    values[8] = np.nan
    _write_map(maps_dir / "holes.fits", values, nside, coordsys="C")
    m = maps.load_map("holes")
    assert m.frame == "icrs" and (m.tile_width, m.max_order) == (2, 0)
    assert sum(int((m.tile(0, f) == 0).sum()) for f in range(12)) == 2


def test_ring_maps_and_unknown_names_are_refused(maps_dir):
    nside = 2
    _write_map(maps_dir / "ring.fits", np.ones(12 * nside * nside, np.float32), nside, ordering="RING")
    with pytest.raises(ValueError, match="NESTED"):
        maps.load_map("ring")
    with pytest.raises(FileNotFoundError):
        maps.load_map("absent")
    with pytest.raises(FileNotFoundError):
        maps.load_map("../etc/passwd")


def test_map_is_rebuilt_only_when_its_file_changes(maps_dir):
    nside = 2
    p = maps_dir / "m.fits"
    _write_map(p, np.full(12 * nside * nside, 1.0, np.float32), nside)
    first = maps.load_map("m")
    assert maps.load_map("m") is first
    p.unlink()
    _write_map(p, np.full(12 * nside * nside, 5.0, np.float32), nside)
    import os
    os.utime(p, (first.mtime + 10, first.mtime + 10))
    assert maps.load_map("m") is not first
    (maps_dir / "remote.hips").write_text("https://example.org/tree\n")
    assert maps.available_maps() == ["m", "remote"]
