"""The share gate refuses what would make the map worse and lets through,
with a warning, what the coadd's weights handle. Pure."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from metadata import share_gate as G

GOOD = dict(analysis={"psf": {"fwhm_arcsec": 3.0, "n_stars": 400}, "noise_px": 1e-4},
            cert={"scale": 0.01, "n_stars": 2000, "sigma": 0.04}, solved=True, n_frames=24,
            concerns=[], statistics={"clipped_high_fraction": 1e-6}, train_fwhm=3.2)


def _codes(r, kind):
    return [x["code"] for x in r[kind]]


def test_a_good_master_is_ready_with_nothing_to_say():
    r = G.readiness(**GOOD)
    assert r == {"ready": True, "blocking": [], "warnings": []}


def test_each_systematic_blocks_and_says_why():
    assert _codes(G.readiness(**{**GOOD, "solved": False}), "blocking") == ["solve"]
    assert _codes(G.readiness(**{**GOOD, "cert": None}), "blocking") == ["flux"]
    assert _codes(G.readiness(**{**GOOD, "cert": {"sigma": 0.3}}), "blocking") == ["flux"]
    assert _codes(G.readiness(**{**GOOD, "analysis": {"psf": {"fwhm_arcsec": None, "n_stars": 4}}}), "blocking") == ["psf"]
    assert _codes(G.readiness(**{**GOOD, "analysis": None}), "blocking") == ["psf"]
    assert _codes(G.readiness(**{**GOOD, "n_frames": 3}), "blocking") == ["frames"]
    no_flat = [{"event": "Stacking with no dark/flat master", "missing": {"dark": 2, "flat": 2}}]
    assert _codes(G.readiness(**{**GOOD, "concerns": no_flat}), "blocking") == ["flat"]
    no_dark = [{"event": "Stacking with no dark master", "missing": {"dark": 2}}]
    assert G.readiness(**{**GOOD, "concerns": no_dark})["ready"]          # a dark is not a flat
    assert _codes(G.readiness(**{**GOOD, "statistics": {"clipped_high_fraction": 0.03}}), "blocking") == ["clipped"]
    r = G.readiness(**{**GOOD, "solved": False, "n_frames": 2})
    assert not r["ready"] and _codes(r, "blocking") == ["solve", "frames"]
    for reason in r["blocking"]:
        assert reason["title"] and len(reason["detail"]) > 40           # a reason a person can act on


def test_what_the_weights_handle_only_warns():
    r = G.readiness(**{**GOOD, "analysis": {"psf": {"fwhm_arcsec": 6.0, "n_stars": 50}}})
    assert r["ready"] and _codes(r, "warnings") == ["blurry"]
    r = G.readiness(**{**GOOD, "analysis": {"psf": {"fwhm_arcsec": 1.5, "n_stars": 50}}})
    assert r["ready"] and _codes(r, "warnings") == ["sharp"]
    r = G.readiness(**{**GOOD, "analysis": {"psf": {"fwhm_arcsec": 6.0, "n_stars": 50}}, "train_fwhm": None})
    assert r["ready"] and r["warnings"] == []                            # no typical PSF yet: nothing to compare to
    r = G.readiness(**{**GOOD, "cert": {"sigma": 0.15}})
    assert r["ready"] and _codes(r, "warnings") == ["flux"]
    r = G.readiness(**{**GOOD, "n_frames": 7})
    assert r["ready"] and _codes(r, "warnings") == ["frames"]
    body = [{"event": "Calibrated with another camera body's flat master", "other_body": {"flat": 1}}]
    r = G.readiness(**{**GOOD, "concerns": body})
    assert r["ready"] and _codes(r, "warnings") == ["body"]


def test_readiness_from_sidecars_uses_the_train_median(tmp_path):
    from file_io.sidecar import set_section
    paths = []
    for i, fwhm in enumerate((3.0, 3.2, 3.1, 6.5)):
        p = tmp_path / f"m{i}.tiff"
        p.write_bytes(b"x")
        set_section(p, "analysis", {"psf": {"fwhm_arcsec": fwhm, "n_stars": 100}})
        set_section(p, "spfc_calibration", {"sigma": 0.05})
        set_section(p, "plate_solution", {"data": {"center": [1, 2]}})
        set_section(p, "image_metadata", {"statistics": {"clipped_high_fraction": 0.0}})
        paths.append(p)
    ui = {f"s{i}": {"focal_length": 351 if i < 3 else 350, "filter": "H", "filenames": ["a"] * 20} for i in range(4)}
    ui["s9"] = {"focal_length": 1234, "filter": "H", "filenames": ["a"] * 2}         # no master yet
    masters = [(f"s{i}", "P", p) for i, p in enumerate(paths)]
    out = G.stacks_readiness(ui, masters, lambda sid: [])
    assert set(out) == {"s0", "s1", "s2", "s3"}
    assert all(out[s]["ready"] for s in out)
    assert _codes(out["s3"], "warnings") == ["blurry"]                   # 6.5″ against the train's 3.15″
    assert out["s0"]["warnings"] == []
    # a master whose sidecar says nothing is blocked on every count it cannot answer
    bare = tmp_path / "bare.tiff"
    bare.write_bytes(b"x")
    r = G.stacks_readiness(ui, [("s0", "P", bare)], lambda sid: [])["s0"]
    assert _codes(r, "blocking") == ["solve", "flux", "psf"]
    assert G.master_inputs(Path(tmp_path / "missing.tiff"))["solved"] is False
    assert np.isfinite(G.MAX_SCATTER)
