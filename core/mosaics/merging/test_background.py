"""Pedestal-recovery tests for the Montage background level matching.

Pure pieces only — the @node wrappers get their integration exercise through
the live pipeline (`hybrid_graxpert_montage_oss_mosaic`). Everything here
runs without Ray, without the app cache, and without Montage installed
(the solver test skips itself when mBgModel is absent).
"""
import os
import shutil
import tempfile

import numpy as np
import pytest
import pyvips

from mosaics.merging.background import (
    _solve_lstsq, _solve_mbgmodel, pairwise_level_fits,
)


def _panel(dirpath: str, name: str, pedestal: float, x0: int,
           W: int = 1600, H: int = 1200, span: int = 900) -> str:
    """A canvas-sized mono float TIFF: sky+pedestal inside [x0, x0+span),
    zero (no-data) outside — the shape `place_image` produces."""
    rng = np.random.default_rng(hash(name) % 2**32)
    canvas = np.zeros((H, W), dtype=np.float32)
    canvas[:, x0:x0 + span] = (
        0.005 + pedestal + rng.normal(0, 2e-4, (H, span)).astype(np.float32))
    p = os.path.join(dirpath, f"{name}.tiff")
    pyvips.Image.new_from_memory(canvas.tobytes(), W, H, 1, "float").tiffsave(p)
    return p


def test_pairwise_fits_recover_known_offsets(tmp_path):
    d = str(tmp_path)
    paths = [_panel(d, "a", 0.0, 0), _panel(d, "b", 0.008, 700)]
    fits = pairwise_level_fits(paths)
    assert len(fits) == 1
    f = fits[0]
    assert (f["plus"], f["minus"]) == (0, 1)
    # c = plus − minus (the lab-verified Montage convention)
    assert abs(f["c"] - (-0.008)) < 5e-4
    assert f["npixel"] > 0 and f["boxwidth"] > 0 and f["boxheight"] > 0


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


def test_no_overlap_yields_no_constraint(tmp_path):
    d = str(tmp_path)
    paths = [_panel(d, "a", 0.0, 0, span=400), _panel(d, "b", 0.01, 1100, span=400)]
    assert pairwise_level_fits(paths) == []


def test_apply_preserves_no_data_zeros(tmp_path):
    """The streamed apply body: subtract where data, keep 0 where none."""
    d = str(tmp_path)
    p = _panel(d, "a", 0.008, 300, span=600)
    src = pyvips.Image.new_from_file(p, access="sequential")
    corrected = (src != 0).ifthenelse(src - 0.008, 0.0)
    out = os.path.join(d, "m.tiff")
    corrected.tiffsave(out)
    a = np.asarray(pyvips.Image.new_from_file(out))
    assert (a[:, :300] == 0).all() and (a[:, 900:] == 0).all(), "no-data must stay 0"
    inside = a[:, 300:900]
    assert abs(np.median(inside) - 0.005) < 5e-4, "sky lands at the shared level"
