"""`calibrate_master` hands GraXpert no NaN: GraXpert can turn a handful of
NaN pixels into a frame of nothing but NaN (MDW field 1347, 2026-10-23).
Holes go in as sky and come back out as holes, and the master carries a
"NaN pixels" concern."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.io import fits

from file_io.generic import read_image
from stacks.post_processing import graxpert_stack


def _graxpert_that_spreads_nan(src, dst, **_):
    a = fits.getdata(src).astype(np.float32)
    fits.writeto(dst, np.full_like(a, np.nan) if np.isnan(a).any() else a - np.median(a))


def test_nan_holes_survive_graxpert_as_holes_and_nothing_else(tmp_path, monkeypatch):
    import graxpert_cli
    import metadata.master_analysis
    import photometry.harvest
    import photometry.solution_check
    monkeypatch.setattr(graxpert_cli, "run_background_extraction", _graxpert_that_spreads_nan)
    monkeypatch.setattr(photometry.harvest, "spfc_scale",
                        lambda *a, **k: (float(graxpert_stack._SPFC_REFERENCE_SCALE), 100, 0.01))
    monkeypatch.setattr(photometry.solution_check, "check_solution",
                        lambda *a: SimpleNamespace(ok=True))
    monkeypatch.setattr(metadata.master_analysis, "measure_master", lambda *a, **k: {"noise_px": 1.0})

    rng = np.random.default_rng(0)
    pixels = (0.01 + 0.001 * rng.standard_normal((64, 64))).astype(np.float32)
    pixels[0, :3] = pixels[30, 40] = np.nan
    src = tmp_path / "master.fits"
    fits.writeto(src, pixels, fits.Header({"FILTER": "Ha"}))
    wcs = {"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRVAL1": 304.0, "CRVAL2": 38.7,
           "CRPIX1": 32.0, "CRPIX2": 32.0, "CDELT1": -0.001, "CDELT2": 0.001}
    master = SimpleNamespace(filename=str(src), plate_solution=SimpleNamespace(header=wcs),
                             stable_hash="ab12cd34ef56ab78", master_integration_s=3600.0)
    config = SimpleNamespace(development_settings=SimpleNamespace(mock_pixinsight=False),
                             background_extraction=True, background_pedestal=0.005)
    out = tmp_path / "out.xisf"

    import concerns
    warned = []
    monkeypatch.setattr(concerns, "warn", lambda title, *a, **k: warned.append((title, k)))
    graxpert_stack.calibrate_master(master, "Ha", "stack ab12", config, out)

    result = np.asarray(read_image(out), np.float32).squeeze()
    assert np.array_equal(np.isnan(result), np.isnan(pixels))
    assert warned == [("NaN pixels", {"nan_pixels": 4})]
