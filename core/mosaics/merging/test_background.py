"""Pedestal-recovery tests for the fused Montage level matching.

Pure pieces plus the fused blend itself (`_blend_many(level_match=True)`)
over real bbox-on-disk panels — no Ray, no app cache; the mBgModel-specific
test skips itself when Montage is absent (solve_levels falls back to the
numpy least-squares solve of the same problem, so the fused test runs
either way).
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

from mosaics.merging.background import _solve_lstsq, _solve_mbgmodel
from mosaics.merging.oss import _blend_many, _level_fit_rows
from mosaics.placement.panel_io import write_placed_panel


def _panel_arrays(pedestal: float, r0: int, r1: int, c0: int, c1: int,
                  seed: int):
    """(bbox, mask, crop) in the blend's retained-crop form."""
    rng = np.random.default_rng(seed)
    crop = (0.005 + pedestal
            + rng.normal(0, 2e-4, (r1 - r0, c1 - c0))).astype(np.float32)
    return (r0, r1, c0, c1), crop > 0, crop


def test_level_fit_rows_recover_known_offset():
    ba, ma, ca = _panel_arrays(0.0, 0, 300, 0, 400, seed=1)
    bb, mb, cb = _panel_arrays(0.008, 0, 300, 250, 650, seed=2)
    rows = _level_fit_rows([ba, bb], [ma, mb], [ca, cb])
    assert len(rows) == 1
    f = rows[0]
    # c = plus - minus (the lab-verified Montage convention)
    assert (f["plus"], f["minus"]) == (0, 1)
    assert abs(f["c"] - (-0.008)) < 5e-4
    # REAL overlap box in canvas units — degenerate boxes make mBgModel
    # silently zero every correction (the 2026-08-19 lab regression).
    assert f["npixel"] == 300 * 150
    assert (f["boxwidth"], f["boxheight"]) == (150.0, 300.0)
    assert (f["xmin"], f["xmax"], f["ymin"], f["ymax"]) == (250, 400, 0, 300)


def test_no_overlap_yields_no_constraint():
    ba, ma, ca = _panel_arrays(0.0, 0, 300, 0, 400, seed=3)
    bb, mb, cb = _panel_arrays(0.01, 0, 300, 900, 1300, seed=4)
    assert _level_fit_rows([ba, bb], [ma, mb], [ca, cb]) == []


def test_tiny_overlap_is_gated():
    """A few-pixel seam cannot carry a level constraint."""
    ba, ma, ca = _panel_arrays(0.0, 0, 10, 0, 400, seed=5)
    bb, mb, cb = _panel_arrays(0.01, 0, 10, 390, 790, seed=6)   # 10x10 px
    assert _level_fit_rows([ba, bb], [ma, mb], [ca, cb]) == []


def test_lstsq_equalises_three_pedestals():
    peds = [0.0, 0.010, -0.005]
    fits = [
        {"plus": 0, "minus": 1, "c": peds[0] - peds[1], "npixel": 25000, "rms": 1e-4},
        {"plus": 1, "minus": 2, "c": peds[1] - peds[2], "npixel": 25000, "rms": 1e-4},
    ]
    levels = _solve_lstsq(fits, 3)
    resid = np.array(peds) - np.array(levels)
    assert resid.max() - resid.min() < 1e-12, "corrected panels must agree exactly"


@pytest.mark.skipif(shutil.which("mBgModel") is None,
                    reason="Montage not installed on this node")
def test_mbgmodel_matches_ground_truth():
    peds = [0.0, 0.010, -0.005]
    box = {"xmin": 2400, "xmax": 3192, "ymin": 0, "ymax": 2384,
           "xcenter": 2796.0, "ycenter": 1192.0,
           "boxwidth": 792.0, "boxheight": 2384.0}
    fits = [
        {"plus": 0, "minus": 1, "c": peds[0] - peds[1],
         "npixel": 1_600_000, "rms": 1.5e-4, **box},
        {"plus": 1, "minus": 2, "c": peds[1] - peds[2],
         "npixel": 1_600_000, "rms": 1.5e-4, **box},
    ]
    levels = _solve_mbgmodel(fits, 3, 8003, 2384)
    resid = np.array(peds) - np.array(levels)
    assert resid.max() - resid.min() < 1e-9, (
        "mBgModel must equalise the pedestals (degenerate-box zeroing "
        "regression: this failed with all-zero corrections before the real "
        "overlap boxes were emitted)")


def _write_panel(dirpath, name: str, pedestal: float, x0: int,
                 span: int, W: int = 1600, H: int = 1200) -> Path:
    """A bbox-on-disk placed panel (the on-disk form the blend reads)."""
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    bbox_image = (0.005 + pedestal
                  + rng.normal(0, 2e-4, (H, span))).astype(np.float32)
    path = Path(dirpath) / f"{name}.tiff"
    write_placed_panel(path, bbox_image, (x0, 0), (W, H))
    return path


def test_fused_blend_removes_pedestal_and_keeps_zeros(tmp_path):
    """The whole point, end to end: two panels at different pedestals in,
    one mosaic with a SINGLE sky level out — and no-data stays exact 0."""
    a = _write_panel(tmp_path, "a", 0.000, x0=100, span=800)   # covers x 100..900
    b = _write_panel(tmp_path, "b", 0.008, x0=700, span=800)   # covers x 700..1500
    plain = _blend_many([a, b])
    matched = _blend_many([a, b], level_match=True)

    # sky regions owned purely by one panel or the other
    sky_a = np.median(matched[:, 150:650])
    sky_b = np.median(matched[:, 950:1450])
    assert abs(sky_a - sky_b) < 5e-4, "levels must agree after matching"
    plain_gap = np.median(plain[:, 950:1450]) - np.median(plain[:, 150:650])
    # The unmatched blend's LOG-SPACE FLUX solve cannot tell pedestal from
    # flux: on this fixture it multiplies panel b by ~0.5 (distorting any
    # real signal!) yet still leaves ~0.0015 of the 0.008 gap. Measured
    # 2026-08-20; the fused match removes the gap to ~1e-6 WITHOUT touching
    # flux (post-rectification scales are ~1).
    assert plain_gap > 1e-3, "sanity: a pedestal gap must survive the plain blend"
    assert abs(sky_a - sky_b) < plain_gap / 10

    # no-data canvas stays exact zero on both sides
    assert (matched[:, :100] == 0).all() and (matched[:, 1500:] == 0).all()
