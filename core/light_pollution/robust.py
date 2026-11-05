"""Robust linear fits, deterministic.

`lmeds`: least median of squares — RANSAC without a threshold — for the
few-parameter fits (a plane over a frame's cells, a frame's scale and offset
against its run): random minimal subsets, keep the one with the smallest
median |residual| (judged on a cell subsample), take its inliers at 2.5 robust
sigma, polish with Tukey-biweight IRLS. Breakdown point 50 %; no scale needed.
A satellite trail, a cloud edge or a star the binning left behind cannot pull it.

`irls` / `irls_bounded`: Tukey-biweight iteratively reweighted least squares
for the larger fits (the site model's shape and amplitudes), the bounded form
with non-negative coefficients: glow and moonlight only add light. With a
`nuisance`, each group of rows has one free coefficient of its own on a
given column (a stack's sky-fixed gradient, scaled per frame), partialled
out under the current weights at every pass (`project_out`): the fit on
what remains is the joint fit, without a column per group.

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


def project_out(A: np.ndarray, y: np.ndarray, w: np.ndarray, groups, omega=None) -> tuple[np.ndarray, np.ndarray]:
    """Frisch–Waugh–Lovell: within each group of rows (index arrays), `A` and `y` minus their
    `omega`-weighted projection on `w`, the row's coefficient on the group's own free parameter.
    Rows in no group, or with w = 0, pass through."""
    A, y = np.array(A, float), np.array(y, float)
    omega = np.ones(len(y)) if omega is None else omega
    for idx in groups:
        ww = omega[idx] * w[idx]
        d = float(ww @ w[idx])
        if d > 0:
            y[idx] -= w[idx] * (ww @ y[idx] / d)
            A[idx] -= np.outer(w[idx], ww @ A[idx] / d)
    return A, y


def _solve(A: np.ndarray, y: np.ndarray, bounded: bool) -> np.ndarray:
    if bounded:
        from scipy.optimize import lsq_linear
        return lsq_linear(A, y, bounds=(0, np.inf)).x
    return np.linalg.lstsq(A, y, rcond=None)[0]


def irls(A: np.ndarray, y: np.ndarray, iters: int = 8, bounded: bool = False, nuisance: tuple | None = None,
         prior: int = 0, scale: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """(coefficients, inlier mask, robust sigma). `bounded`: every coefficient >= 0.
    `nuisance` = (w, groups) as `project_out` takes them. The last `prior` rows are a prior:
    weight 1 always, outside the robust scale — which is `scale` when given, else re-estimated
    from the inliers each pass."""
    data = np.arange(len(y)) < len(y) - prior
    project = (lambda omega: project_out(A, y, *nuisance, omega)) if nuisance else (lambda omega: (A, y))
    Ap, yp = project(None)
    coef = _solve(Ap, yp, bounded)
    inl = np.ones(len(y), bool)

    def sigma(r):
        if scale is not None:
            return scale
        keep = inl & data
        return mad_scale(r[keep]) if keep.any() else mad_scale(r[data])

    for _ in range(iters):
        r = yp - Ap @ coef
        wt = np.where(data, _weights(r, sigma(r)), 1.0)
        if wt.sum() < A.shape[1] + 2:
            break
        Ap, yp = project(wt)
        sw = np.sqrt(wt)
        new = _solve(Ap * sw[:, None], yp * sw, bounded)
        done = np.allclose(new, coef, rtol=1e-4, atol=1e-9)
        coef, inl = new, wt > 0
        if done:
            break
    return coef, inl, sigma(yp - Ap @ coef)


def irls_bounded(A: np.ndarray, y: np.ndarray, iters: int = 8, **kw):
    """`irls` with every coefficient >= 0."""
    return irls(A, y, iters, bounded=True, **kw)


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
