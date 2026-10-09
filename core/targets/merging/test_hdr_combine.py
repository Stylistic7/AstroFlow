"""The PixInsight-free HDR combine (`targets.merging.oss.hdr_combine`) on a
synthetic star field: a short master at 0.4x the long one's flux on a
different sky level, offset by a fractional shift and aligned only through
the two plate solutions, with a star clipped in the long exposure and intact
in the short one."""
import numpy as np
import pytest
from astropy.wcs import WCS

from targets.merging.oss import hdr_combine

SHAPE = (300, 400)
SHIFT = (3.4, -2.1)                      # (dx, dy) of the short master
BRIGHT = (200.0, 150.0, 2.0)             # x, y, peak: clipped in the long master
CLIP = 0.8                               # the long master's saturation level
BG = 0.005                               # the pinned background pedestal
RATIO = 0.4                              # short / long flux (120 s vs 300 s)


def _wcs(dx: float = 0.0, dy: float = 0.0) -> WCS:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [10.0, 20.0]
    w.wcs.crpix = [200.0 + dx, 150.0 + dy]
    w.wcs.cdelt = [-1 / 3600, 1 / 3600]
    return w


def _signal(dx: float = 0.0, dy: float = 0.0, n: int = 80) -> np.ndarray:
    """Stars above the pedestal, in the long master's flux units."""
    rng = np.random.default_rng(7)
    stars = [(rng.uniform(25, SHAPE[1] - 25), rng.uniform(25, SHAPE[0] - 25), rng.uniform(0.05, 0.4))
             for _ in range(n)] + [BRIGHT]
    y, x = np.mgrid[0:SHAPE[0], 0:SHAPE[1]].astype(np.float64)
    img = np.zeros(SHAPE)
    for sx, sy, peak in stars:
        img += peak * np.exp(-((x - sx - dx) ** 2 + (y - sy - dy) ** 2) / (2 * 1.8 ** 2))
    return img


def test_short_exposure_is_rescaled_restores_the_clipped_core_and_noise_drops():
    rng = np.random.default_rng(1)
    s_long, s_short = 0.002, 0.0015          # the short master's own units
    long = np.minimum(BG + _signal() + rng.normal(0, s_long, SHAPE), CLIP).astype(np.float32)
    short = (3 * BG + RATIO * _signal(*SHIFT) + rng.normal(0, s_short, SHAPE)).astype(np.float32)
    out, report = hdr_combine([long, short], [_wcs(), _wcs(*SHIFT)], [s_long, s_short])

    truth = BG + _signal()
    x, y = int(BRIGHT[0]), int(BRIGHT[1])
    assert long[y, x] == CLIP                                       # it really was clipped
    assert abs(out[y, x] - truth[y, x]) < 0.08, (out[y, x], truth[y, x])
    assert abs(report["masters"][1]["scale_to_reference"] - RATIO) < 0.02, report
    # Inverse-variance mean of the long data and the rescaled short data
    # (its noise grows by 1/RATIO), on a star-free patch.
    patch = (slice(2, 22), slice(2, 60))
    expected = 1 / np.sqrt(1 / s_long ** 2 + (RATIO / s_short) ** 2)
    assert np.std(out[patch] - truth[patch]) < 1.3 * expected
    assert np.std(out[patch] - truth[patch]) < 0.9 * s_long
    assert np.isfinite(out).all()


def test_uncovered_pixels_take_the_covering_masters_only():
    """The short master, shifted +3.4 px, does not cover the reference's
    right edge: those pixels are the long master alone (never a mean with a
    zero fill); where it covers, the mean differs from the long master."""
    rng = np.random.default_rng(2)
    long = (BG + _signal() + rng.normal(0, 0.002, SHAPE)).astype(np.float32)
    short = (BG + _signal(*SHIFT) + rng.normal(0, 0.002, SHAPE)).astype(np.float32)
    out, report = hdr_combine([long, short], [_wcs(), _wcs(*SHIFT)], [0.002, 0.002])
    right, left = (slice(20, 60), slice(397, 400)), (slice(20, 60), slice(5, 40))
    assert np.allclose(out[right], long[right], rtol=0, atol=1e-7)
    assert not np.allclose(out[left], long[left], atol=1e-4)
    assert report["masters"][1]["covered_fraction"] < 1.0


def test_nothing_bright_in_common_refuses():
    blank = np.full(SHAPE, BG, np.float32) + np.random.default_rng(3).normal(0, 0.002, SHAPE).astype(np.float32)
    with pytest.raises(ValueError, match="bright pixels in common"):
        hdr_combine([blank, blank.copy()], [None, None], [0.002, 0.002])
