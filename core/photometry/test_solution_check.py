"""The plate-solution check against REAL pixels: the M42 testdata sub with
its own solution passes; the same solution with a false cubic radial
distortion added — the shape a second-order SIP leaves behind on a reducer
(2026-09-20) — fails, and the consumers act on the verdict: the sub harvest
excludes the frame with a concern, the master calibration raises.
"""
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from photometry import solution_check as sc
from photometry.gaia_catalog import FILTERS, GaiaCatalog, write_catalog

SUB = Path(__file__).resolve().parents[1] / "testdata/input/Light_M42_180.0s_Bin1_2600MM_20231130-233344_0021.fit"


@pytest.fixture(scope="module")
def frame(tmp_path_factory):
    """The real sub's pixels + a catalogue MADE FROM ITS OWN STARS through a
    plain TAN projection at the header's pointing and pixel scale: the
    catalogue then reproduces the peaks EXACTLY under that projection, and
    every departure from it is the test's own doing. (No solution is
    involved — the frame's real optics are not the truth here.)"""
    d = tmp_path_factory.mktemp("m42")
    img = fits.getdata(SUB).astype(np.float32)
    h = fits.getheader(SUB)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [float(h["RA"]), float(h["DEC"])]
    wcs.wcs.crpix = [img.shape[1] / 2 + 0.5, img.shape[0] / 2 + 0.5]
    scale = 206.265 * float(h["XPIXSZ"]) / float(h["FOCALLEN"]) / 3600.0
    wcs.wcs.cdelt = [-scale, scale]
    peaks = sc._peaks(img)                                     # bright local maxima, (x, y) 1-based
    keep = (peaks[:, 0] > 40) & (peaks[:, 0] < img.shape[1] - 40) & (peaks[:, 1] > 40) & (peaks[:, 1] < img.shape[0] - 40)
    peaks = peaks[keep]
    # brightest ~600 by the image itself, so the catalogue looks like the bright end of Gaia
    v = img[(peaks[:, 1] - 1).astype(int), (peaks[:, 0] - 1).astype(int)]
    peaks = peaks[np.argsort(-v)[:600]]
    sky = wcs.all_pix2world(peaks, 1)
    n = len(sky); order = np.argsort(sky[:, 1])
    write_catalog(d / "cat", dict(source_id=np.arange(n), ra=sky[order, 0].astype(np.float64), dec=sky[order, 1].astype(np.float64),
                                  G=np.full(n, 10.0, np.float32), colour=np.full(n, 1.0, np.float32),
                                  flux=np.ones((n, len(FILTERS)), np.float32)))
    return img, wcs, GaiaCatalog(d / "cat")


def _with_radial_distortion(wcs: WCS, k3: float) -> WCS:
    """`wcs` plus a cubic radial displacement about CRPIX, applied as SIP:
    x' = x + k3 * x * r^2 (A_3_0 = A_1_2 = k3, B_0_3 = B_2_1 = k3)."""
    h = wcs.to_header(relax=True)
    h["A_ORDER"] = h["B_ORDER"] = 3
    for k in list(h.keys()):
        if k.startswith(("A_", "B_", "AP_", "BP_")) and k not in ("A_ORDER", "B_ORDER"):
            del h[k]
    h["A_3_0"] = h["A_1_2"] = k3
    h["B_0_3"] = h["B_2_1"] = k3
    h["CTYPE1"], h["CTYPE2"] = "RA---TAN-SIP", "DEC--TAN-SIP"
    return WCS(h)


def test_the_real_solution_places_its_stars(frame, monkeypatch):
    img, wcs, cat = frame
    monkeypatch.setattr("photometry.harvest.catalog", lambda: cat)
    v = sc.check_solution(img, wcs, "H")
    assert v.n_stars > 200 and v.max_shift_px <= sc.MAX_OFFSET_PX and v.ok, str(v)


def test_the_threshold_belongs_to_the_caller(frame, monkeypatch):
    """A master's WCS places the image in a mosaic, so it is held to one
    pixel. A sub's only seeds apertures, and a badly seeded aperture excludes
    itself — so a sub is held to two, and contributes the cells that worked
    instead of being discarded whole."""
    img, wcs, cat = frame
    monkeypatch.setattr("photometry.harvest.catalog", lambda: cat)
    v = sc.check_solution(img, _with_radial_distortion(wcs, 6e-11), "H")
    assert sc.MASTER_MAX_OFFSET_PX < v.max_shift_px <= sc.SUB_MAX_OFFSET_PX, str(v)
    assert v.ok_within(sc.SUB_MAX_OFFSET_PX) and not v.ok_within(sc.MASTER_MAX_OFFSET_PX)
    assert not v.ok                              # the bare property is the master's rule


def test_a_cubic_radial_error_fails_the_check(frame, monkeypatch):
    """k3 = 4e-10: 0.4 px at r=1000, 12 px at r=3000 — the corner error a
    quadratic SIP leaves on the 0.63"/px train, scaled to this frame."""
    img, wcs, cat = frame
    monkeypatch.setattr("photometry.harvest.catalog", lambda: cat)
    bad = _with_radial_distortion(wcs, 4e-10)
    v = sc.check_solution(img, bad, "H")
    assert not v.ok and v.max_shift_px > 3.0 and v.worst_region.startswith("cell"), str(v)
    # a tiny term — inside the tolerance everywhere — still passes
    ok = sc.check_solution(img, _with_radial_distortion(wcs, 2e-11), "H")
    assert ok.ok, str(ok)


def test_forced_photometry_rejects_stars_the_solution_misplaces(frame, monkeypatch):
    """The centroid step is bounded to one pixel: a star seeded 3 px off is
    NOT found and measured faint — it is flagged."""
    from photometry.forced_photometry import CENTROID_STEP_PX, measure
    img, wcs, cat = frame
    st = cat.stars_in_field(wcs, img.shape[1], img.shape[0], "H")
    right = measure(img / 65535.0, st["x"], st["y"])
    off = measure(img / 65535.0, st["x"] + 3.0, st["y"])
    # seeded exactly: the centroid barely moves (M42's saturated cores and
    # nebula-buried peaks fail OTHER gates; the step gate passes nearly all)
    assert np.median(right["offset"]) < 0.3 and (right["offset"] > CENTROID_STEP_PX).mean() < 0.02
    # 3 px off: flagged wherever the star's core is what the window sees (a
    # seed held by M42's nebulosity barely moves and is not — the solution
    # check, not this gate, is what refuses a systematic offset)
    assert off["ok"].mean() < 0.4
    assert np.median(off["offset"][np.isfinite(off["offset"])]) > sc.MAX_OFFSET_PX


def test_sub_harvest_excludes_a_bad_solution_with_a_concern(frame, monkeypatch, tmp_path):
    from file_io.sidecar import get_section, set_section
    from metadata import star_photometry as sp
    img, wcs, cat = frame
    monkeypatch.setattr("photometry.harvest.catalog", lambda: cat)
    p = tmp_path / "sub.fit"
    fits.PrimaryHDU(img.astype(np.uint16), header=fits.Header({"FILTER": "H"})).writeto(p)
    bad = _with_radial_distortion(wcs, 4e-10)
    set_section(p, "plate_solution", {"schema": "t", "data": {"header": dict(bad.to_header(relax=True))}})
    warned = []
    monkeypatch.setattr("concerns.warn", lambda title, detail="", **f: warned.append((title, f)))
    assert sp.compute_star_photometry(p) is None
    marker = get_section(p, sp.SECTION)
    assert marker["reason"] == sp._BAD_SOLUTION and marker["shift_px"] > 3.0
    assert warned and "excluded" in warned[0][0]
    # the verdict is remembered for THIS solution only: re-solving retries
    assert sp.compute_star_photometry(p) is None and len(warned) == 1
    set_section(p, "plate_solution", {"schema": "t", "data": {"header": dict(wcs.to_header(relax=True))}})
    monkeypatch.setattr("photometry.harvest.harvest_sub_xp", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("reached the harvest")))
    with pytest.raises(RuntimeError, match="reached the harvest"):
        sp.compute_star_photometry(p)


def test_master_calibration_raises_on_a_bad_solution(frame, monkeypatch, tmp_path):
    """`post_stack_calibrate` (the GraXpert route's gain calibration) refuses
    a master whose solution fails the check, naming the geometry — before
    any photometry runs."""
    from stacks.post_processing import graxpert_stack as gs
    img, wcs, cat = frame
    monkeypatch.setattr("photometry.harvest.catalog", lambda: cat)
    bad = _with_radial_distortion(wcs, 4e-10)
    master = tmp_path / "m.fits"
    fits.PrimaryHDU(img).writeto(master)

    from decisions.settings import DevelopmentSettings, ProcessingConfig
    from metadata.image_metadata import compute_image_metadata
    from metadata.sidecar_io import write_image_metadata
    from plate_solving import PlateSolution
    from stacks.composite_stack import Stack
    import dataclasses
    meta = compute_image_metadata._function(master, plate_solve=False)
    meta = dataclasses.replace(meta, plate_solution=PlateSolution(
        header={k: v for k, v in bad.to_header(relax=True).items()}, num_stars=100,
        center=(100.0, 45.0), image_scale=0.37, rotation_angle=0.0))
    write_image_metadata(meta)
    from testing.test_stacks import make_composite_stack
    stack = make_composite_stack(filter="H")
    config = ProcessingConfig(development_settings=DevelopmentSettings(mock_pixinsight=False))
    monkeypatch.setattr(gs.post_stack_calibrate, "filename_fn", lambda *a: str(tmp_path / "out" / "m.xisf"), raising=False)
    monkeypatch.setattr("photometry.harvest.spfc_scale", lambda *a, **k: (_ for _ in ()).throw(AssertionError("photometry must not run")))
    with pytest.raises(ValueError, match="does not place the stars"):
        gs.post_stack_calibrate._function(stack, config, meta)


def test_the_gaia_refit_repairs_a_distorted_solution(frame, monkeypatch):
    """The solver's cubic left unconstrained past an index-tile edge is a
    smooth error a few pixels deep at the edge (LBN 522 OIII, 2026-10-14:
    3.5 px in one corner). Refitting against the catalogue stars actually in
    the image recovers the true projection everywhere."""
    img, wcs, cat = frame
    monkeypatch.setattr("photometry.harvest.catalog", lambda: cat)
    bad = _with_radial_distortion(wcs, 1.5e-10)
    assert not sc.check_solution(img, bad, "H").ok
    fixed = sc.refit(img, bad, "H")
    v = sc.check_solution(img, fixed, "H")
    assert v.ok and v.max_shift_px < 0.5, str(v)


def _solution(wcs: WCS):
    from plate_solving import PlateSolution
    return PlateSolution(header=dict(wcs.to_header(relax=True)), num_stars=600,
                         center=(0.0, 0.0), image_scale=1.0, rotation_angle=0.0)


def test_a_solution_that_passes_is_never_refit(frame, monkeypatch):
    import plate_solving
    img, wcs, cat = frame
    monkeypatch.setattr("photometry.harvest.catalog", lambda: cat)
    good = _solution(wcs)
    assert plate_solving._gaia_checked(SUB, good) is good


def test_a_failing_solution_is_replaced_by_the_refit(frame, monkeypatch):
    import plate_solving
    img, wcs, cat = frame
    monkeypatch.setattr("photometry.harvest.catalog", lambda: cat)
    out = plate_solving._gaia_checked(SUB, _solution(_with_radial_distortion(wcs, 1.5e-10)))
    from astropy.wcs import WCS as _W
    from astropy.io.fits import Header
    v = sc.check_solution(img, _W(Header(out.header)), "H")
    assert v.ok and v.max_shift_px < 0.5, str(v)
    mid = wcs.pixel_to_world((img.shape[1] - 1) / 2, (img.shape[0] - 1) / 2)
    assert abs(out.center[0] - mid.ra.deg) * 3600 < 1 and abs(out.center[1] - mid.dec.deg) * 3600 < 1


def test_the_check_needs_no_filter(frame, monkeypatch):
    """Geometry only: a produced image whose FILTER says 'none' (a BXT'd
    channel TIFF, 2026-10-14) is checked and refit like any other."""
    import plate_solving
    img, wcs, cat = frame
    monkeypatch.setattr("photometry.harvest.catalog", lambda: cat)
    good = _solution(wcs)
    assert plate_solving._gaia_checked(SUB, good) is good
