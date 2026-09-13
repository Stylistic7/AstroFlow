"""The per-stack star flat gain: fit on synthetic subs, and the arithmetic.

A synthetic sensor prints a smooth 1% pattern; subs are sky x pattern^k_true
plus a pedestal, noise and stars. The fit must recover k_true when the dark
(pedestal) is subtracted, read LOW without it (the documented dilution), and
refuse to believe an absurd answer.
"""
import numpy as np
import pytest

from stacks.star_flat_gain import (
    GAIN_RANGE, apply_gain, block_mean, fit_gain, sample_paths,
)

H, W = 2000, 2800        # a small detector: room for the band's edge margin
SCALE = 8


def _pattern(rng, amp=0.012):
    """A smooth random field with ~300-600 px blobs, unit-normalised."""
    from scipy.ndimage import gaussian_filter
    f = gaussian_filter(rng.standard_normal((H, W)), 60)
    f -= f.mean(); f /= f.std()
    return 1.0 + amp * f


def _subs(rng, flat, k_true, n=8, pedestal=800.0, sky=3000.0, stars=200):
    out = {}
    yy, xx = np.mgrid[:H, :W]
    for i in range(n):
        s = sky * (1 + 0.1 * (xx / W) + 0.05 * (yy / H) + 0.02 * i)      # a sky gradient, drifting per sub
        img = s * (1.0 + k_true * (flat - 1.0)) + pedestal
        for _ in range(stars):                                             # bright points, per-sub positions
            y, x = rng.integers(2, H - 2), rng.integers(2, W - 2)
            img[y - 1:y + 2, x - 1:x + 2] += rng.uniform(5, 60) * sky
        img += rng.normal(0, 0.01 * sky, img.shape)
        out[f"sub{i}"] = img.astype(np.float32)
    return out


@pytest.mark.parametrize("k_true", [0.8, 1.4])
def test_fit_recovers_the_gain_when_the_pedestal_is_removed(k_true):
    rng = np.random.default_rng(1)
    flat = _pattern(rng)
    subs = _subs(rng, flat, k_true)
    dark = np.full((H, W), 800.0, np.float32)
    fit = fit_gain(list(subs), flat, dark, scale=SCALE, read=subs.__getitem__)
    assert fit["n_used"] == len(subs) and "reason" not in fit
    assert abs(fit["k"] - k_true) < 0.08, fit
    assert abs(np.median(fit["corrs"])) > 0.6      # the sign follows k - 1


def test_fit_reads_low_without_the_dark():
    """An un-subtracted pedestal both dilutes the slope and prints an inverted
    copy of the pattern: slope = ((k - 1) sky - pedestal) / (sky + pedestal)
    = (0.4 * 3000 - 800) / 3800, so a 1.4 gain reads ~1.1."""
    rng = np.random.default_rng(2)
    flat = _pattern(rng)
    subs = _subs(rng, flat, 1.4)
    with_dark = fit_gain(list(subs), flat, np.full((H, W), 800.0, np.float32), scale=SCALE, read=subs.__getitem__)["k"]
    without = fit_gain(list(subs), flat, None, scale=SCALE, read=subs.__getitem__)["k"]
    assert without < with_dark
    assert abs(without - (1 + (0.4 * 3000 - 800) / 3800)) < 0.06


def test_fit_refuses_an_absurd_gain_and_too_few_subs():
    rng = np.random.default_rng(3)
    flat = _pattern(rng)
    subs = _subs(rng, flat, 4.0, n=4)                                   # far outside GAIN_RANGE
    fit = fit_gain(list(subs), flat, np.full((H, W), 800.0, np.float32), scale=SCALE, read=subs.__getitem__)
    assert fit["k"] == 1.0 and "outside" in fit["reason"] and fit["k_fit"] > GAIN_RANGE[1]
    two = fit_gain(list(subs)[:2], flat, None, scale=SCALE, read=subs.__getitem__)
    assert two["k"] == 1.0 and "only 2" in two["reason"]


def test_apply_gain_scales_the_pattern_about_its_median():
    rng = np.random.default_rng(4)
    flat = _pattern(rng).astype(np.float32) * 0.5                        # a master flat's 0.5 level
    unit = apply_gain(flat, 1.0)
    assert unit.dtype == np.float32 and abs(np.median(unit) - 1.0) < 1e-3
    twice = apply_gain(flat, 2.0)
    np.testing.assert_allclose(twice - 1.0, 2.0 * (unit - 1.0), atol=1e-6)


def test_sample_paths_spans_the_list_evenly():
    paths = [f"p{i}" for i in range(100)]
    picked = sample_paths(paths, 10)
    assert len(picked) == 10 and picked[0] == "p0" and picked[-1] == "p99"
    assert sample_paths(paths[:5], 10) == paths[:5]


def test_block_mean_crops_to_a_multiple():
    a = np.arange(7 * 9, dtype=float).reshape(7, 9)
    b = block_mean(a, 4)
    assert b.shape == (1, 2) and b[0, 0] == a[:4, :4].mean()
