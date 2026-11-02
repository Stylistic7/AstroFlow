"""Robust linear fits, deterministic.

`lmeds`: least median of squares — RANSAC without a threshold — for the
few-parameter fits (a plane over a frame's cells, a frame's scale and offset
against its run): random minimal subsets, keep the one with the smallest
median |residual| (judged on a cell subsample), take its inliers at 2.5 robust
sigma, polish with Tukey-biweight IRLS. Breakdown point 50 %; no scale needed.
A satellite trail, a cloud edge or a star the binning left behind cannot pull it.

`irls` / `irls_bounded`: Tukey-biweight iteratively reweighted least squares
for the larger fits (the site model's shape and amplitudes), the bounded form
with non-negative coefficients: glow and moonlight only add light.

Every fit seeds its own generator, so a refit is bit-identical whatever ran
before it — a model's identity is a hash of its numbers.
"""
import numpy as np

TUKEY_C = 4.685


def mad_scale(r: np.ndarray) -> float:
    m = np.median(r)
    return float(1.4826 * np.median(np.abs(r - m)) + 1e-12)


def _weights(r: np.ndarray, s: float) -> np.ndarray:
    u = r / (TUKEY_C * s)
    return np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)


def irls(A: np.ndarray, y: np.ndarray, iters: int = 8) -> tuple[np.ndarray, np.ndarray, float]:
    """(coefficients, inlier mask, robust sigma)."""
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    inl = np.ones(len(y), bool)
    for _ in range(iters):
        r = y - A @ coef
        wt = _weights(r, mad_scale(r[inl]) if inl.any() else mad_scale(r))
        if wt.sum() < A.shape[1] + 2:
            break
        sw = np.sqrt(wt)
        new = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
        done = np.allclose(new, coef, rtol=1e-4, atol=1e-9)
        coef, inl = new, wt > 0
        if done:
            break
    r = y - A @ coef
    return coef, inl, mad_scale(r[inl]) if inl.any() else mad_scale(r)


def irls_bounded(A: np.ndarray, y: np.ndarray, iters: int = 8) -> tuple[np.ndarray, np.ndarray, float]:
    """`irls` with every coefficient >= 0."""
    from scipy.optimize import lsq_linear
    bounds = (0, np.inf)
    coef = lsq_linear(A, y, bounds=bounds).x
    inl = np.ones(len(y), bool)
    for _ in range(iters):
        r = y - A @ coef
        wt = _weights(r, mad_scale(r[inl]) if inl.any() else mad_scale(r))
        if wt.sum() < A.shape[1] + 2:
            break
        sw = np.sqrt(wt)
        new = lsq_linear(A * sw[:, None], y * sw, bounds=bounds).x
        done = np.allclose(new, coef, rtol=1e-4, atol=1e-9)
        coef, inl = new, wt > 0
        if done:
            break
    r = y - A @ coef
    return coef, inl, mad_scale(r[inl]) if inl.any() else mad_scale(r)


def lmeds(A: np.ndarray, y: np.ndarray, trials: int | None = None, sample: int = 5000,
          seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """(coefficients, inlier mask, robust sigma)."""
    n, p = A.shape
    if n < max(p + 2, 10):
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        return coef, np.ones(n, bool), mad_scale(y - A @ coef)
    rng = np.random.default_rng(seed)
    trials = trials or {1: 30, 2: 60, 3: 120}.get(p, 250)
    sub = rng.choice(n, min(sample, n), replace=False)
    As, ys = A[sub], y[sub]
    best, best_med = None, np.inf
    for _ in range(trials):
        idx = rng.choice(n, p, replace=False)
        try:
            coef = np.linalg.solve(A[idx], y[idx])
        except np.linalg.LinAlgError:
            continue
        med = np.median(np.abs(ys - As @ coef))
        if med < best_med:
            best, best_med = coef, med
    if best is None:
        best = np.linalg.lstsq(A, y, rcond=None)[0]
        best_med = np.median(np.abs(y - A @ best))
    s0 = 1.4826 * best_med * (1 + 5 / max(n - p, 1))
    inl = np.abs(y - A @ best) < 2.5 * s0
    if inl.sum() < p + 2:
        inl[:] = True
    coef, w, s = irls(A[inl], y[inl])
    full = np.zeros(n, bool)
    full[np.flatnonzero(inl)[w]] = True
    return coef, full, s
