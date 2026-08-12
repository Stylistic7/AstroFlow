"""Unit tests for the pure-numpy STF autostretch (the palette formulas moved
into the live-finish stdlib — see finishes/test_finishes.py)."""
import numpy as np

from mosaics.color.palettes import autostretch


def test_autostretch_maps_median_to_target_and_bounds():
    rng = np.random.default_rng(0)
    x = np.clip(0.10 + 0.02 * rng.standard_normal((256, 256)), 0, 1).astype(np.float32)
    out = autostretch(x, target_background=0.15)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.all(np.isfinite(out))
    # median is preserved by the monotonic MTF, so it lands on the target
    assert abs(float(np.median(out)) - 0.15) < 0.02


def test_autostretch_handles_empty_and_constant_input():
    assert np.all(autostretch(np.full((8, 8), np.nan, np.float32)) == 0.0)
    flat = autostretch(np.full((8, 8), 0.2, np.float32))
    assert np.all(np.isfinite(flat)) and flat.min() >= 0.0 and flat.max() <= 1.0
