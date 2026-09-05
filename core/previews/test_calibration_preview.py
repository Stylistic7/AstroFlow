"""calibrate_image — preview-grade partial calibration.

Regression for 2026-09-04: the unified CALIBRATION_TOPOLOGY gives lights
`("dark", "flat")` and never a bias, but calibrate_image required bias AND
dark AND flat, silently returning the raw light otherwise. Every
flat-corrected Run preview therefore served the raw frame — byte-identical
carousel entries for every candidate flat, a flat-picker that picked
nothing. These tests pin the partial-calibration semantics.
"""
import numpy as np
from types import SimpleNamespace

from previews.calibration import calibrate_image


def _md(exposure):
    return SimpleNamespace(header={"EXPOSURE": exposure}, filename="test")


def _fixtures():
    rng = np.random.default_rng(3)
    H, W = 120, 160
    light = (0.2 + rng.random((H, W)) * 0.1).astype(np.float32)
    dark = (0.01 + rng.random((H, W)) * 0.002).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    vig = 1.0 - 0.4 * (((yy - H / 2) / H) ** 2 + ((xx - W / 2) / W) ** 2)
    flat_a = (vig * 0.5).astype(np.float32)
    flat_b = flat_a.copy()
    flat_b[40:60, 60:80] *= 0.7          # dust patch — the thing the picker exists to show
    return light, dark, flat_a, flat_b


def test_no_bias_still_applies_dark_and_flat():
    """The production topology: lights carry dark+flat, no bias. The flat
    MUST be applied — this exact case used to return the raw light."""
    light, dark, flat_a, _ = _fixtures()
    out = calibrate_image(_md(300), light, None, None,
                          _md(300), dark, _md(3), flat_a, False)
    assert np.abs(out - light).max() > 0.01, "calibration was a no-op"


def test_different_flats_produce_different_previews():
    """The carousel's whole purpose: candidate flats must render
    distinguishably (byte-identical entries = the 2026-09-04 bug)."""
    light, dark, flat_a, flat_b = _fixtures()
    args = (_md(300), light, None, None, _md(300), dark)
    out_a = calibrate_image(*args, _md(3), flat_a, False)
    out_b = calibrate_image(*args, _md(3), flat_b, False)
    assert np.abs(out_a - out_b).max() > 0.01
    # and the dust patch is where they differ
    diff = np.abs(out_a - out_b)
    assert diff[40:60, 60:80].mean() > 10 * diff[:20, :20].mean()


def test_no_flat_returns_uncorrected():
    """The 2-arg "uncorrected" preview path: no flat → the light as-is."""
    light, dark, *_ = _fixtures()
    out = calibrate_image(_md(300), light, None, None,
                          None, None, None, None, False)
    assert np.allclose(out, light)


def test_flat_only_no_dark():
    """A run with a flat but no dark still gets the flat division."""
    light, _, flat_a, _ = _fixtures()
    out = calibrate_image(_md(300), light, None, None,
                          None, None, _md(3), flat_a, False)
    assert np.abs(out - light).max() > 0.01


def test_rotation_still_applies():
    light, *_ = _fixtures()
    out = calibrate_image(_md(300), light, None, None,
                          None, None, None, None, True)
    assert np.allclose(out, np.rot90(light, 2))
