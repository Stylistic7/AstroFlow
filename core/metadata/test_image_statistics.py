"""Statistics correctness across the exact (small-image) and subsampled
(large-image) paths of `compute_image_statistics`.

The large-image subsampling exists to stop `compute_image_metadata` from
peaking ~35 GB (float64 upcast + full-array sorts) when it's chained after a
gigapixel produced master/mosaic. These tests pin the two invariants that make
that safe: small images are computed exactly as before, and the sampled path
tracks the exact answer closely while keeping the scalar reductions exact.
"""
import numpy as np
import pytest

import metadata.image_statistics as ims
from metadata.image_statistics import compute_image_statistics


def _stats_exact(image):
    """Force the full-array path regardless of size."""
    old = ims._MAX_EXACT_PIXELS
    ims._MAX_EXACT_PIXELS = image.size + 1
    try:
        return compute_image_statistics(image)
    finally:
        ims._MAX_EXACT_PIXELS = old


def _stats_sampled(image, sample):
    """Force the subsample path with a given sample size."""
    old_max, old_n = ims._MAX_EXACT_PIXELS, ims._SAMPLE_PIXELS
    ims._MAX_EXACT_PIXELS, ims._SAMPLE_PIXELS = 0, sample
    try:
        return compute_image_statistics(image)
    finally:
        ims._MAX_EXACT_PIXELS, ims._SAMPLE_PIXELS = old_max, old_n


def _synthetic(n, frac_zero=0.4, seed=1):
    """Sky-dominated log-normal + a sparse bright-star tail + zero sentinels —
    the shape a real placed panel / mosaic has."""
    rng = np.random.default_rng(seed)
    vals = rng.lognormal(mean=0.0, sigma=0.5, size=n).astype(np.float32)
    vals[rng.random(n) < 0.001] += 50.0          # rare saturated stars
    vals[rng.random(n) < frac_zero] = 0.0        # no-data sentinel
    return vals.reshape(1000, n // 1000)


def test_exact_path_known_values():
    """The preserved full-array path still produces the textbook answers."""
    img = np.array([[0, 1, 2, 3, 4, 100]], np.float32)  # one zero sentinel
    s = _stats_exact(img)
    assert s.min_value == 1.0                     # zero dropped
    assert s.max_value == 100.0
    assert s.mean_value == pytest.approx((1 + 2 + 3 + 4 + 100) / 5)
    assert s.p50 == pytest.approx(3.0)
    assert sum(s.histogram_counts) == 5           # five non-zero pixels


def test_sampling_approximates_exact():
    img = _synthetic(1_000_000)
    exact = _stats_exact(img)
    samp = _stats_sampled(img, sample=100_000)

    # max & mean are taken over the FULL array in the sample path -> exact.
    assert samp.max_value == exact.max_value
    assert samp.mean_value == pytest.approx(exact.mean_value, rel=1e-9)

    # Percentiles are estimated from the sample -> close, not exact.
    for attr in ("p1", "p5", "p25", "p50", "p75", "p95", "p99"):
        assert getattr(samp, attr) == pytest.approx(
            getattr(exact, attr), rel=0.03, abs=0.05
        ), attr
    assert samp.sky_median == pytest.approx(exact.sky_median, rel=0.05, abs=0.05)

    # Histogram counts are rescaled back to full-image totals.
    assert sum(samp.histogram_counts) == pytest.approx(
        sum(exact.histogram_counts), rel=0.02
    )


def test_sampling_is_deterministic():
    img = _synthetic(500_000, seed=7)
    assert _stats_sampled(img, sample=50_000) == _stats_sampled(img, sample=50_000)


def test_all_zero_image_is_degenerate_not_a_crash():
    img = np.zeros((2000, 2000), np.float32)
    s = _stats_sampled(img, sample=10_000)   # count_scale hits the /0 guard
    assert s.max_value == 0.0 and s.min_value == 0.0
    assert sum(s.histogram_counts) == 0
