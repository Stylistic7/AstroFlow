"""Forced photometry on synthetic stars, the passband weights, and the
catalogue field lookup on a synthetic WCS."""
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from photometry import forced_photometry as fp
from photometry.gaia_catalog import FILTERS, WAVELENGTHS_NM, _band_weights, band_fluxes, write_catalog


def _frame(rng, n=40, H=600, W=800, fwhm=2.0, bg=0.02, noise=0.0005):
    img = np.full((H, W), bg, np.float32) + rng.normal(0, noise, (H, W)).astype(np.float32)
    # normalised units: peaks well below the brightest star's, SNR above 40
    x = rng.uniform(40, W - 40, n); y = rng.uniform(40, H - 40, n); flux = rng.uniform(0.2, 1.5, n)
    sig = fwhm / 2.3548; yy, xx = np.mgrid[:H, :W]
    for xi, yi, fi in zip(x, y, flux):     # FITS 1-based centres: pixel index = coord - 1
        img += (fi / (2 * np.pi * sig ** 2) * np.exp(-((xx - (xi - 1)) ** 2 + (yy - (yi - 1)) ** 2) / (2 * sig ** 2))).astype(np.float32)
    return img, x, y, flux


def test_measure_recovers_flux_centroid_fwhm():
    rng = np.random.default_rng(0)
    img, x, y, flux = _frame(rng)
    # a bright, unsaturated reference star sets the frame's saturation threshold
    # (PI's rule: 0.75 x the brightest clean star's peak); it is itself rejected
    sig = 2.0 / 2.3548; yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    img += (4.0 / (2 * np.pi * sig ** 2) * np.exp(-((xx - 699) ** 2 + (yy - 499) ** 2) / (2 * sig ** 2))).astype(np.float32)
    m = fp.measure(img, np.r_[x + rng.normal(0, 0.4, len(x)), 700.0], np.r_[y + rng.normal(0, 0.4, len(y)), 500.0])   # start ~0.5 px off
    assert m["ok"][:-1].all() and not m["ok"][-1]
    assert m["saturation"] == pytest.approx(0.75 * m["peak"][-1], rel=1e-6)
    m = {k: (v[:-1] if isinstance(v, np.ndarray) else v) for k, v in m.items()}
    # PI convention: 0-based with centres at +0.5 == FITS 1-based - 0.5
    assert np.allclose(m["x"], x - 0.5, atol=0.05) and np.allclose(m["y"], y - 0.5, atol=0.05)
    # aperture r = 2*FWHM+1.5 = 5.5 px on a 2 px FWHM Gaussian holds 99.99% of the flux
    assert np.allclose(m["signal"] / flux, 1.0, atol=0.03)      # faintest stars carry ~2% photon noise
    assert np.allclose(m["fwhm"] / 2.0, 1 / 1.148 * 1.0, rtol=0.08)   # moment FWHM x calibration
    assert (m["snr"] > 40).all()


def test_measure_flags_edges_and_saturation():
    rng = np.random.default_rng(1)
    img, x, y, flux = _frame(rng, n=5)
    img[100:104, 100:104] = 0.9                       # a blob far above 0.75 x the brightest clean star
    m = fp.measure(img, np.r_[x, 102.5, 3.0], np.r_[y, 102.5, 3.0])
    assert m["ok"][:5].all() and not m["ok"][5] and not m["ok"][6]
    assert m["saturation"] == pytest.approx(0.75 * 0.9, rel=0.02)   # the blob itself is the brightest unsaturated peak


def test_band_weights_are_passband_integrals():
    W = _band_weights()
    flat = np.ones(343)
    assert W["H"] @ flat == pytest.approx(3.0, rel=1e-6)              # 3 nm box on a flat spectrum
    assert np.count_nonzero(W["H"]) <= 4 and WAVELENGTHS_NM[np.argmax(W["H"])] == 656.0
    assert 0 < W["R"] @ flat < W["L"] @ flat < 2.0 * 343                # L = envelope of R/G/B, not the flat ideal QE
    assert np.all(W["L"] >= W["R"]) and np.all(W["L"] >= W["B"])
    f = band_fluxes(np.vstack([flat, 2 * flat]))
    assert f.shape == (2, len(FILTERS)) and np.allclose(f[1], 2 * f[0])


def test_stars_in_field_projects_the_frame_footprint(tmp_path):
    from photometry.gaia_catalog import GaiaCatalog
    n = 500; rng = np.random.default_rng(2)
    dec = np.sort(rng.uniform(-3, 3, n)); ra = rng.uniform(177, 183, n)
    write_catalog(tmp_path / "cat", dict(source_id=np.arange(n), ra=ra, dec=dec,
                  G=np.full(n, 10.0, np.float32), colour=np.full(n, 1.0, np.float32),
                  flux=np.ones((n, len(FILTERS)), np.float32)))
    cat = GaiaCatalog(tmp_path / "cat")
    w = WCS(fits.Header({"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRVAL1": 180.0, "CRVAL2": 0.0,
                         "CRPIX1": 500.5, "CRPIX2": 300.5, "CD1_1": -0.002, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": 0.002}))
    st = cat.stars_in_field(w, 1000, 600, "H")
    assert 10 < len(st["x"]) < n
    # frame = 2.0 x 1.2 deg minus the 16 px margin (0.032 deg) on each side
    inside = (np.abs(ra - 180) < 1.0 - 0.032) & (np.abs(dec) < 0.6 - 0.032)
    assert len(st["x"]) == pytest.approx(inside.sum(), abs=3)
    assert np.allclose(st["colour"], 1.0) and st["catalog_flux"].shape == st["x"].shape


def test_synthetic_photometry_scales():
    from photometry.gaia_catalog import synthetic_photometry
    flat = np.ones((1, 343), np.float32)
    g1, c1 = synthetic_photometry(flat); g2, c2 = synthetic_photometry(100 * flat)
    assert g2[0] == pytest.approx(g1[0] - 5.0, abs=1e-4)          # x100 flux = 5 mag brighter
    assert c1[0] == pytest.approx(c2[0]) and np.isfinite(c1[0])    # colour is a flux ratio
    g0, c0 = synthetic_photometry(np.zeros((1, 343), np.float32))
    assert np.isnan(g0[0]) and np.isnan(c0[0])


def test_harvest_gate_accepts_a_correct_wcs_and_rejects_a_wrong_one(tmp_path):
    """`harvest_sub_xp` against a stand-in catalogue: stars measured at their
    catalogue positions give one tight scale (good); the same catalogue 8 px
    off — a wrong plate solution — must not pass the gate."""
    from file_io.sidecar import set_section
    from photometry import harvest

    rng = np.random.default_rng(1)
    H, W, n, sigma = 600, 800, 300, 2.5 / 2.355
    # a jittered 40 px grid: no blends, so every bright star measures the same scale
    col, row = np.meshgrid(np.arange(20), np.arange(15))
    x = (col.ravel() + 0.5) * 40 + rng.uniform(-8, 8, n)
    y = (row.ravel() + 0.5) * 40 + rng.uniform(-8, 8, n)
    G = rng.uniform(8.5, 13.0, n)
    flux = 0.3 * 10 ** (-0.4 * (G - 8.0))                       # sub-saturation for the brightest
    yy, xx = np.mgrid[:H, :W]
    image = np.full((H, W), 0.02) + rng.normal(0, 1e-4, (H, W))
    for xi, yi, fi in zip(x, y, flux):
        sl = (slice(int(yi) - 8, int(yi) + 9), slice(int(xi) - 8, int(xi) + 9))
        image[sl] += fi / (2 * np.pi * sigma**2) * np.exp(-((xx[sl] - xi) ** 2 + (yy[sl] - yi) ** 2) / (2 * sigma**2))
    p = tmp_path / "sub.fit"
    fits.PrimaryHDU((image * 65535).clip(0, 65535).astype(np.uint16)).writeto(p)
    w = WCS(naxis=2)
    w.wcs.crpix, w.wcs.cdelt, w.wcs.crval, w.wcs.ctype = [W / 2, H / 2], [-1 / 3600, 1 / 3600], [100, 45], ["RA---TAN", "DEC--TAN"]
    set_section(p, "plate_solution", {"schema": "test", "data": {"header": dict(w.to_header())}})

    class Catalogue:
        def __init__(self, dx):
            self.dx = dx

        def stars_in_field(self, wcs, width, height, key):
            assert (width, height, key) == (W, H, "H")
            return {"source_id": np.arange(n), "x": x + self.dx, "y": y, "ra": x, "dec": y, "G": G,
                    "colour": rng.uniform(0.5, 1.5, n), "catalog_flux": flux}

    stars, info = harvest.harvest_sub_xp(p, filter="H", cat=Catalogue(0.0))
    # the window's scatter is set by the faint end at the SNR cut, not the engine
    assert info["good"] and info["n_window"] > 100 and info["scatter"] < 0.1
    bright = stars[:, 4] < 10.5
    assert stars.shape[1] == 10 and np.std(np.log(stars[bright, 8])) < 0.04    # scale = signal / catalogue flux
                                                                                # (noise-limited: 3.8% at G 10.5)
    _, wrong = harvest.harvest_sub_xp(p, filter="H", cat=Catalogue(8.0))
    assert not wrong["good"]


def _synthetic_field(rng, H=600, W=800, n=300, fwhm=2.5):
    """A jittered grid of unblended stars with known total flux, plus a
    stand-in catalogue that reports those fluxes. Shared by the spfc_scale
    tests below."""
    sigma = fwhm / 2.355
    col, row = np.meshgrid(np.arange(20), np.arange(15))
    x = (col.ravel() + 0.5) * 40 + rng.uniform(-8, 8, n)
    y = (row.ravel() + 0.5) * 40 + rng.uniform(-8, 8, n)
    G = rng.uniform(8.5, 13.0, n)
    flux = 0.3 * 10 ** (-0.4 * (G - 8.0))
    yy, xx = np.mgrid[:H, :W]
    image = np.full((H, W), 0.02) + rng.normal(0, 1e-4, (H, W))
    for xi, yi, fi in zip(x, y, flux):
        sl = (slice(int(yi) - 8, int(yi) + 9), slice(int(xi) - 8, int(xi) + 9))
        image[sl] += fi / (2 * np.pi * sigma**2) * np.exp(
            -((xx[sl] - xi) ** 2 + (yy[sl] - yi) ** 2) / (2 * sigma**2))
    w = WCS(naxis=2)
    w.wcs.crpix, w.wcs.cdelt = [W / 2, H / 2], [-1 / 3600, 1 / 3600]
    w.wcs.crval, w.wcs.ctype = [100, 45], ["RA---TAN", "DEC--TAN"]
    colour = rng.uniform(0.5, 1.5, n)

    class Catalogue:
        def __init__(self, dx=0.0):
            self.dx = dx

        def stars_in_field(self, wcs, width, height, key):
            return {"source_id": np.arange(n), "x": x + self.dx, "y": y,
                    "ra": x, "dec": y, "G": G, "colour": colour,
                    "catalog_flux": flux}

    return image.astype(np.float32), w, Catalogue


def test_spfc_scale_is_linear_in_image_brightness():
    """THE property the gain depends on. `post_stack_calibrate` multiplies by
    `SPFC_REFERENCE_SCALE / scale`, so a master that is k times brighter must
    report k times the scale — otherwise the gain does not cancel it and every
    panel lands on a different photometric gauge."""
    from photometry.harvest import spfc_scale

    rng = np.random.default_rng(3)
    image, w, Catalogue = _synthetic_field(rng)
    s1, n1, _ = spfc_scale(image, w, filter="H", cat=Catalogue())
    s2, n2, _ = spfc_scale(image * 2.0, w, filter="H", cat=Catalogue())
    assert n1 > 100 and n2 > 100
    assert s2 / s1 == pytest.approx(2.0, rel=0.02)

    # and therefore the gain-corrected pixels are invariant
    from units import SPFC_REFERENCE_SCALE
    assert (image * (SPFC_REFERENCE_SCALE / s1)).mean() == pytest.approx(
        (image * 2.0 * (SPFC_REFERENCE_SCALE / s2)).mean(), rel=0.02)


def test_spfc_scale_agrees_with_the_per_star_scale_column():
    """`spfc_scale` aggregates exactly the ratio `harvest_sub_xp` stores per
    star, so the frame number must sit on the median of that column."""
    from photometry.forced_photometry import measure
    from photometry.harvest import MIN_SNR, spfc_scale

    rng = np.random.default_rng(4)
    image, w, Catalogue = _synthetic_field(rng)
    cat = Catalogue()
    scale, n, sigma = spfc_scale(image, w, filter="H", cat=cat)

    st = cat.stars_in_field(w, image.shape[1], image.shape[0], "H")
    m = measure(image, st["x"], st["y"])
    ok = m["ok"] & (st["catalog_flux"] > 0) & (m["snr"] >= MIN_SNR)
    per_star = m["signal"][ok] / st["catalog_flux"][ok]
    win = (st["G"][ok] >= 8.0) & (st["G"][ok] <= 13.5)
    assert scale == pytest.approx(float(np.median(per_star[win])), rel=1e-6)
    assert sigma > 0 and n == int(win.sum())


def test_spfc_scale_refuses_a_wrong_wcs():
    """A wrong plate solution measures noise at every catalogue position. That
    must RAISE, not return a plausible number — an uncalibrated master passed
    downstream is the failure this guards (PI's "Insufficient data" did the
    same job)."""
    from photometry.harvest import spfc_scale

    rng = np.random.default_rng(5)
    image, w, Catalogue = _synthetic_field(rng)
    with pytest.raises(ValueError, match="too few calibratable stars"):
        spfc_scale(image, w, filter="H", cat=Catalogue(dx=8.0))


def test_spfc_scale_needs_a_filter():
    from photometry.harvest import spfc_scale

    rng = np.random.default_rng(6)
    image, w, Catalogue = _synthetic_field(rng)
    with pytest.raises(ValueError, match="no FITS FILTER"):
        spfc_scale(image, w, filter=None, cat=Catalogue())
