"""The star_photometry sidecar section: harvest-once caching, the two
`data: None` markers (provisional vs final), and the gather into the star
flat's layout. The photometry engine itself is mocked (needs the catalogue)."""
import numpy as np
import pytest
import astropy.io.fits as pf

from file_io.sidecar import get_section, set_section
from photometry.gaia_catalog import CATALOG_PATH
from metadata import star_photometry as sp
from stacks.starflat import gather_star_photometry

pytestmark = pytest.mark.skipif(not CATALOG_PATH.exists(),
                                reason="the Gaia XP catalogue is not at ASTROFLOW_GAIA_CATALOG")

_STARS = np.array([
    # x, y, ra, dec, G, colour, fwhm, signal, scale, snr
    [10.123, 20.456, 0, 0, 9.0, 0.5, 1.61, 12345.6, 1.02, 250.04],
    [30.0, 40.0, 0, 0, 11.5, 1.3, 1.55, 999.0, 0.98, 41.0],
    [50.0, 60.0, 0, 0, 10.0, np.nan, 1.5, 100.0, 1.00, 80.0],   # NaN colour: dropped
])
_GOOD = {"engine": "astroflow-xp", "n_window": 2, "scatter": 0.03, "gross": 0.0, "good": True}


@pytest.fixture
def sub(tmp_path):
    p = tmp_path / "Light_test.fit"
    pf.PrimaryHDU(np.zeros((4, 4), np.uint16), header=pf.Header({"FILTER": "Ha"})).writeto(p)
    return p


def _solve(p):
    set_section(p, "plate_solution",
                {"schema": "astroflow.platesolution.v2", "data": {"header": {"CRPIX2": 2.0}}})


def test_harvests_once_and_reads_back(sub, monkeypatch):
    calls = []
    monkeypatch.setattr("photometry.harvest.harvest_sub_xp",
                        lambda *a, **k: calls.append(k) or (_STARS, dict(_GOOD)))
    _solve(sub)
    t = sp.compute_star_photometry(sub)
    assert t.shape == (2, 8)
    assert t[0].tolist() == [10.12, 20.46, 9.0, 0.5, 1.61, 12345.6, 1.02, 250.0]
    assert calls[0]["filter"] == "Ha"
    d = get_section(sub, "star_photometry")["data"]
    assert d["n"] == 2 and d["engine"] == "astroflow-xp"
    # cached: a second call never touches SPFC
    assert np.array_equal(sp.compute_star_photometry(sub), t) and len(calls) == 1
    assert np.array_equal(sp.read_star_photometry(sub), t)


def test_no_solution_marker_is_provisional(sub, monkeypatch):
    calls = []
    monkeypatch.setattr("photometry.harvest.harvest_sub_xp",
                        lambda *a, **k: calls.append(1) or (_STARS, dict(_GOOD)))
    assert sp.compute_star_photometry(sub) is None and not calls
    assert get_section(sub, "star_photometry")["reason"] == "no_plate_solution"
    assert sp.read_star_photometry(sub) is None
    _solve(sub)                                    # the solve arrives later
    assert sp.compute_star_photometry(sub).shape == (2, 8) and len(calls) == 1


def test_no_match_marker_is_final(sub, monkeypatch):
    calls = []
    monkeypatch.setattr("photometry.harvest.harvest_sub_xp",
                        lambda *a, **k: calls.append(1) or (_STARS, {**_GOOD, "gross": 0.9, "good": False}))
    _solve(sub)
    assert sp.compute_star_photometry(sub) is None
    m = get_section(sub, "star_photometry")
    assert m["data"] is None and m["reason"] == "no_match" and m["gross"] == 0.9
    assert sp.compute_star_photometry(sub) is None and len(calls) == 1


def test_transient_failure_leaves_no_section(sub, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("slot timeout")
    monkeypatch.setattr("photometry.harvest.harvest_sub_xp", boom)
    _solve(sub)
    with pytest.raises(RuntimeError):
        sp.compute_star_photometry(sub)
    assert get_section(sub, "star_photometry") is None


def test_gather_keeps_sub_index(sub, tmp_path):
    sp.write_star_photometry(sub, _STARS, {})
    bare = tmp_path / "Light_bare.fit"
    pf.PrimaryHDU(np.zeros((4, 4), np.uint16)).writeto(bare)
    stars = gather_star_photometry([bare, sub])
    assert stars.shape == (2, 11)
    assert set(stars[:, 10]) == {1.0}                    # the harvested sub is index 1
    assert stars[0, 5] == pytest.approx(0.5)             # colour column
