"""Canonical offline render: the viewer's exact autostretch + Hubble math.

Transcribed from the WebGPU sources — finish_seed.js (STF seed: black =
median − 2.8σ, white = p99, midtones solved so the median lands on the
target grey) and finishes/stock.py `hubble` (R=S, G=H, B=O with
average-neutral SCNR). Lab renders must go through THIS, never a
hand-rolled stretch (standing instruction, 2026-08-31).
"""
from __future__ import annotations

import numpy as np


def _solve_midtones(x0: float, target: float) -> float:
    denom = 2.0 * target * x0 - target - x0
    if abs(denom) < 1e-12:
        return 0.5
    return float(np.clip((x0 * (target - 1.0)) / denom, 1e-4, 0.999))


def _mtf(m: float, x: np.ndarray) -> np.ndarray:
    return ((m - 1.0) * x) / (((2.0 * m - 1.0) * x) - m)


def autostretch_channel(channel: np.ndarray, target: float = 0.15) -> np.ndarray:
    """finish_seed.js `channelStf` on tiles.py-style stats (nonzero pixels,
    σ = 1.4826·MAD), then the MTF — one channel, seeded exactly as the
    viewer seeds it."""
    valid = channel[channel > 0]
    med = float(np.median(valid))
    mad = float(np.median(np.abs(valid - med)))
    sigma = 1.4826 * mad if mad > 0 else max(med * 1e-3, 1e-6)
    bp = max(0.0, med - 2.8 * sigma)
    wp = max(float(np.percentile(valid, 99.0)), bp + 1e-3)
    x0 = (med - bp) / max(wp - bp, 1e-6)
    mid = _solve_midtones(x0, target)
    x = np.clip((channel - bp) / max(wp - bp, 1e-6), 0.0, 1.0)
    return _mtf(mid, x)


def hubble_render(s: np.ndarray, h: np.ndarray, o: np.ndarray,
                  scnr: float = 0.9, target: float = 0.15) -> np.ndarray:
    """stock.hubble minus the (absent) screened-star layer: per-channel
    autostretch, R=S G=H B=O, average-neutral SCNR. Returns uint8 HxWx3."""
    ss = autostretch_channel(s, target)
    hh = autostretch_channel(h, target)
    oo = autostretch_channel(o, target)
    hh = np.minimum(hh, (1.0 - scnr) * hh + scnr * 0.5 * (ss + oo))
    rgb = np.stack([ss, hh, oo], axis=-1)
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def save_png(path, rgb_u8: np.ndarray) -> None:
    import cv2

    cv2.imwrite(str(path), rgb_u8[..., ::-1])
