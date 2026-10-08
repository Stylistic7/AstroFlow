"""How much more of an image more hours would make visible.

A master is signal + noise. More hours leave the signal alone and shrink the
noise as 1/sqrt(t), so from the master alone we can predict which structure
will cross from "lost in the noise" to "clearly there". Structure is seen at
a scale, not per pixel: averaging k x k pixels gains ~k in SNR, so each scale
is tested with its OWN measured noise (registration correlates the noise, so
k is not assumed). A pixel is visible when, at any scale, its block's signal
is at least THRESHOLD times that scale's noise.

    visible_fraction(img, factor)   the fraction of the field visible after
                                    `factor` times the current integration
    gain_per_hour(img, hours, dh)   the fraction newly revealed per added hour

Limits, said plainly: "visible" here is SNR, not perception — denoising
(NXT) and the stretch change what the eye sees, so THRESHOLD wants
calibrating against Stylistic7's processing; and a background model is needed
to stop a sky gradient counting as structure, which also removes nebulosity
larger than BACKGROUND_PX.
"""
from __future__ import annotations

import numpy as np

SCALES = (1, 2, 4, 8)
THRESHOLD = 3.0
BACKGROUND_PX = 256


def _bin(a: np.ndarray, k: int) -> np.ndarray:
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def _noise(a: np.ndarray) -> float:
    """Gaussian σ of `a`'s noise: the robust spread of a 5x5 local-median
    high-pass. NOT neighbour differences — registration interpolation
    correlates neighbouring pixels in a master, so differences read the
    noise low and noise passed as structure (2026-10-08: a simulated
    half-depth master "showed" more than the full one)."""
    from scipy.ndimage import median_filter
    r = (a - median_filter(a, size=5)).ravel()
    r = r[np.isfinite(r)]
    return float(1.4826 * np.median(np.abs(r - np.median(r))) * np.sqrt(25 / 24))


def _background(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """A smooth sky model: the median of each BACKGROUND_PX block, bilinearly
    interpolated back to full size. Not a low percentile: one sits below the
    sky by ~0.8σ, and at coarse scales that offset reads as structure."""
    from scipy.ndimage import zoom
    k = BACKGROUND_PX
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    blocks = np.where(valid, a, np.nan)[:h, :w].reshape(h // k, k, w // k, k).transpose(0, 2, 1, 3).reshape(h // k, w // k, -1)
    grid = np.nanmedian(blocks, axis=2)
    grid = np.where(np.isfinite(grid), grid, np.nanmedian(grid))
    return zoom(grid, (a.shape[0] / grid.shape[0], a.shape[1] / grid.shape[1]), order=1)


def snr_maps(img: np.ndarray) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
    """(valid mask, [(scale, signal/noise per block)]) for the image as it is."""
    a = np.asarray(img, np.float64).squeeze()
    valid = np.isfinite(a) & (a > 0)
    s = np.where(valid, a - _background(a, valid), 0.0)
    out = []
    for k in SCALES:
        b = _bin(s, k)
        out.append((k, b / _noise(b)))
    return valid, out


def visible_fraction(valid: np.ndarray, maps, factor: float) -> float:
    """Fraction of the valid field visible after `factor` x the integration."""
    gain = np.sqrt(factor)
    vis = np.zeros(valid.shape, bool)
    for k, snr in maps:
        hit = np.repeat(np.repeat(snr * gain >= THRESHOLD, k, axis=0), k, axis=1)
        vis[:hit.shape[0], :hit.shape[1]] |= hit
    return float(np.mean(vis[valid]))


def gain_per_hour(valid, maps, hours: float, added: float = 5.0) -> float:
    """Fraction of the field newly visible per hour over the next `added` hours."""
    return (visible_fraction(valid, maps, (hours + added) / hours) - visible_fraction(valid, maps, 1.0)) / added
