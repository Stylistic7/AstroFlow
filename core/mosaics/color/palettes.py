"""Pure-numpy STF autostretch (PixInsight AutoSTF reference math).

The palette formulas that once lived here moved into the live-finish
stdlib (finishes/stdlib.py, finishes/stock.py) when baked palettes were
removed (2026-08-12); what remains is the deterministic autostretch and
its MTF/midtones-solver internals, used by previews and as the reference
in PixInsight-comparison tests. Ray-free and PixInsight-free.
"""
from __future__ import annotations

import numpy as np


def _mtf(x: np.ndarray, midtones: float) -> np.ndarray:
    """PixInsight midtones transfer function MTF(m, x), element-wise.

    MTF(m, x) = (m-1)·x / ((2m-1)·x − m), with the exact endpoints 0→0, 1→1 and
    the identity m=0.5. Lower midtones brighten."""
    if midtones == 0.5:
        return x.astype(np.float32, copy=True)
    num = (midtones - 1.0) * x
    den = (2.0 * midtones - 1.0) * x - midtones
    out = np.where(x <= 0.0, 0.0, np.where(x >= 1.0, 1.0, num / den))
    return out.astype(np.float32)


def _solve_midtones(x0: float, target: float) -> float:
    """The midtones `m` with MTF(m, x0) == target (closed form)."""
    denom = 2.0 * target * x0 - target - x0
    if abs(denom) < 1e-12:
        return 0.5
    return float(np.clip(x0 * (target - 1.0) / denom, 1e-4, 1.0 - 1e-4))


def autostretch(
    image: np.ndarray, *, target_background: float = 0.15, shadows_clip: float = -2.8,
) -> np.ndarray:
    """STF-equivalent autostretch (PixInsight AutoSTF): clip shadows at
    `median + shadows_clip·1.4826·MAD`, then apply the MTF that maps the
    (clipped, renormalized) median to `target_background`. Returns float32 in
    [0,1]. A deterministic *starting* stretch — the user re-stretches to taste.
    """
    x = image.astype(np.float32)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x)
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    sigma = 1.4826 * mad if mad > 0 else 1e-6

    c0 = float(np.clip(med + shadows_clip * sigma, 0.0, 1.0))
    denom = max(1.0 - c0, 1e-6)
    y = np.clip((x - c0) / denom, 0.0, 1.0)
    med_norm = float(np.clip((med - c0) / denom, 1e-6, 1.0 - 1e-6))
    return _mtf(y, _solve_midtones(med_norm, target_background))
