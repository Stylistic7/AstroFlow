"""Pure — no Ray, no context."""
import numpy as np

from light_pollution.robust import irls, irls_bounded, lmeds


def _plane_with_outliers(seed=1, n=20000, outliers=0.3):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, (n, 2))
    A = np.column_stack([np.ones(n), X])
    y = 0.5 + 0.3 * X[:, 0] - 0.2 * X[:, 1] + rng.normal(0, 0.4, n)
    k = int(outliers * n)
    y[:k] += rng.uniform(5, 50, k)          # one-sided gross outliers: a cloud edge, a trail
    return A, y


def test_lmeds_ignores_thirty_percent_gross_outliers():
    A, y = _plane_with_outliers()
    coef, inliers, sigma = lmeds(A, y)
    assert np.allclose(coef, [0.5, 0.3, -0.2], atol=0.02)
    assert 0.6 < inliers.mean() < 0.75
    assert abs(sigma - 0.4) < 0.05
    plain = np.linalg.lstsq(A, y, rcond=None)[0]
    assert abs(plain[0] - 0.5) > 5          # what least squares makes of the same data


def test_irls_recovers_the_same_plane():
    A, y = _plane_with_outliers()
    coef, inliers, _ = irls(A, y)
    assert np.allclose(coef, [0.5, 0.3, -0.2], atol=0.02)


def test_fits_are_deterministic():
    A, y = _plane_with_outliers(seed=7)
    assert np.array_equal(lmeds(A, y)[0], lmeds(A, y)[0])


def test_bounded_fit_keeps_coefficients_non_negative():
    rng = np.random.default_rng(3)
    A = rng.uniform(0, 1, (5000, 2))
    y = 0.7 * A[:, 0] + rng.normal(0, 0.05, 5000) - 0.03 * A[:, 1]   # the second column slightly subtracts
    coef, _, _ = irls_bounded(A, y)
    assert coef[1] < 1e-9 and abs(coef[0] - 0.7) < 0.05
    assert np.linalg.lstsq(A, y, rcond=None)[0][1] < 0                 # which least squares would report


def test_tiny_samples_fall_back_to_least_squares():
    A = np.column_stack([np.ones(5), np.arange(5.0)])
    y = 1 + 2 * np.arange(5.0)
    coef, inliers, _ = lmeds(A, y)
    assert np.allclose(coef, [1, 2]) and inliers.all()
