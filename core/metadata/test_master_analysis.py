"""`analyze_master` measures visibility on a master and stores it in the
sidecar; more hours never reveal less, and noise is not mistaken for
structure. Runs as a script too."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from metadata import visibility as V
from metadata.master_analysis import FACTORS


def _field(rng, sigma, faint=3e-4, h=600, w=800):
    """Flat sky, a faint broad blob and a brighter knot, plus noise."""
    yy, xx = np.mgrid[0:h, 0:w]
    blob = faint * np.exp(-(((xx - 500) / 120) ** 2 + ((yy - 300) / 90) ** 2))
    knot = 10 * faint * np.exp(-(((xx - 250) / 15) ** 2 + ((yy - 200) / 15) ** 2))
    return (0.005 + blob + knot + rng.normal(0, sigma, (h, w))).astype(np.float32)


def test_more_hours_reveal_more_and_noise_is_not_structure():
    rng = np.random.default_rng(1)
    deep, shallow = _field(rng, 2e-4), _field(rng, 4e-4)
    dv, dm = V.snr_maps(deep)
    sv, sm = V.snr_maps(shallow)
    fractions = [V.visible_fraction(dv, dm, f) for f in FACTORS]
    assert all(b >= a for a, b in zip(fractions, fractions[1:]))
    assert V.visible_fraction(sv, sm, 1.0) < V.visible_fraction(dv, dm, 1.0)
    # quality levels nest: what is clean is also usable and detected, and
    # deeper data lifts every level
    levels = V.quality_fractions(dv, dm, 1.0)
    assert levels == sorted(levels, reverse=True)
    assert all(b >= a for a, b in zip(levels, V.quality_fractions(dv, dm, 4.0)))
    empty = (0.005 + rng.normal(0, 2e-4, (600, 800))).astype(np.float32)
    ev, em = V.snr_maps(empty)
    assert V.visible_fraction(ev, em, 1.0) < 0.02


def test_analyze_master_writes_the_sidecar_section(tmp_path):
    from file_io.sidecar import get_section
    from file_io.tiff_file_io import write_tiff_image
    from metadata.master_analysis import SECTION, analyze_master
    path = tmp_path / "m.tiff"
    write_tiff_image(path, _field(np.random.default_rng(2), 2e-4), {"FILTER": "O", "STBLHASH": "abc"})

    class _M:
        filename, stable_hash, header, master_integration_s = path, "abc", {"FILTER": "O"}, 36000.0
    out = inspect.unwrap(analyze_master._function)(_M())
    assert get_section(path, SECTION) == out
    assert out["stack"] == "abc" and out["filter"] == "O"
    curves = out["visibility"]["fraction"]
    assert out["visibility"]["thresholds"] == list(V.THRESHOLDS)
    assert len(curves) == len(V.THRESHOLDS) and all(len(c) == len(FACTORS) for c in curves)
    assert all(b >= a for c in curves for a, b in zip(c, c[1:]))    # more hours never reveal less
    assert out["noise_px"] == pytest.approx(2e-4, rel=0.15) and out["pixel_arcsec"] is None
    assert out["psf"] == {"fwhm_px": None, "n_stars": out["psf"]["n_stars"], "fwhm_arcsec": None}   # a starless field has no PSF to measure


def test_measure_master_reports_the_noise_and_the_psf():
    """What the coadd weights by: the pixels' noise and the PSF's Gaussian
    width, on the sky when the master is solved."""
    from metadata.master_analysis import measure_master
    rng = np.random.default_rng(5)
    img = _field(rng, 2e-4)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    s = 3.5 / 2.3548                                         # 60 stars of FWHM 3.5 px, well above the sky
    for _ in range(60):
        x, y = rng.uniform(40, 760), rng.uniform(40, 560)
        img += np.float32(0.2) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s)).astype(np.float32)
    out = measure_master(img, stable_hash="abc", filter="H", integration_s=3600.0, pixel_arcsec=2.0)
    assert out["version"] == 3 and out["noise_px"] == pytest.approx(2e-4, rel=0.15)
    assert out["psf"]["n_stars"] >= 40
    assert out["psf"]["fwhm_px"] == pytest.approx(3.5, rel=0.1)
    assert out["psf"]["fwhm_arcsec"] == pytest.approx(7.0, rel=0.1)
    assert measure_master(img, stable_hash="abc", filter="H", integration_s=None, pixel_arcsec=None)["psf"]["fwhm_arcsec"] is None


if __name__ == "__main__":
    import tempfile
    test_more_hours_reveal_more_and_noise_is_not_structure(); print("ok more hours / noise")
    test_analyze_master_writes_the_sidecar_section(Path(tempfile.mkdtemp())); print("ok sidecar section")
    test_measure_master_reports_the_noise_and_the_psf(); print("ok noise and psf")
