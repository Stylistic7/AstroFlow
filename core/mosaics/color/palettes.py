"""Pure-numpy color math for the OSS mosaic color pipeline.

An STF-equivalent autostretch (so the dynamic palette weights are meaningful on
otherwise-linear data), the Hubble and Foraxx narrowband palettes, RGB-star
combination, and a screen blend. Everything operates on float arrays; mono
inputs are (H, W), color outputs are (H, W, 3). Kept Ray-free and PixInsight-
free so the math is unit-testable on its own.
"""
from __future__ import annotations

import numpy as np
import pyvips


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


def hubble(sii: np.ndarray, ha: np.ndarray, oiii: np.ndarray) -> np.ndarray:
    """Classic Hubble (SHO) palette: R=SII, G=Hα, B=OIII. Inputs stretched [0,1]."""
    return np.clip(np.dstack([sii, ha, oiii]), 0.0, 1.0).astype(np.float32)


def foraxx(sii: np.ndarray, ha: np.ndarray, oiii: np.ndarray) -> np.ndarray:
    """Foraxx dynamic narrowband palette (ForaxX / thecoldestnights).

    `~x = 1-x`, `^` = power. On stretched, brightness-matched channels:
        R = (O3^~O3)·S2 + ~(O3^~O3)·Ha
        G = ((O3·Ha)^~(O3·Ha))·Ha + ~((O3·Ha)^~(O3·Ha))·O3
        B = O3
    The data-dependent weights let pixel signal decide each channel's mix.
    """
    s = np.clip(sii, 0.0, 1.0).astype(np.float32)
    h = np.clip(ha, 0.0, 1.0).astype(np.float32)
    o = np.clip(oiii, 0.0, 1.0).astype(np.float32)

    wR = o ** (1.0 - o)
    R = wR * s + (1.0 - wR) * h
    m = o * h
    wG = m ** (1.0 - m)
    G = wG * h + (1.0 - wG) * o
    B = o
    return np.clip(np.dstack([R, G, B]), 0.0, 1.0).astype(np.float32)


def combine_star_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Stack the three (already unscreened) mono star channels into one RGB
    star image."""
    return np.clip(np.dstack([r, g, b]), 0.0, 1.0).astype(np.float32)


def screen(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Photoshop 'screen' blend: out = 1 − (1−base)·(1−overlay). Commutative,
    >= max(base, overlay), stays in [0,1]. Used to lay unscreened stars back
    over a starless palette."""
    base = np.clip(base, 0.0, 1.0).astype(np.float32)
    overlay = np.clip(overlay, 0.0, 1.0).astype(np.float32)
    return (1.0 - (1.0 - base) * (1.0 - overlay)).astype(np.float32)


# --- pyvips ports (streaming, memory-bounded) -------------------------------
# The @node deliverables run these on full-res mosaic channels (a single 2.25
# Gpx RGB image is 27 GB), so the palette math must not materialize whole
# arrays. pyvips composes these lazily and evaluates them tile-by-tile in the
# `tiffsave` sink. They mirror the numpy functions above byte-for-byte (element-
# wise ops + band-joins); `test_palettes` asserts the equivalence, so the numpy
# versions above stay the readable, tested spec.


def _clip01(img: pyvips.Image) -> pyvips.Image:
    """Clamp a vips image to [0,1] (vips has no direct clip)."""
    return (img < 0).ifthenelse(0, (img > 1).ifthenelse(1, img))


def hubble_vips(sii: pyvips.Image, ha: pyvips.Image, oiii: pyvips.Image) -> pyvips.Image:
    """Vips port of `hubble`: R=SII, G=Hα, B=OIII."""
    return _clip01(sii.bandjoin([ha, oiii]))


def foraxx_vips(sii: pyvips.Image, ha: pyvips.Image, oiii: pyvips.Image) -> pyvips.Image:
    """Vips port of `foraxx` (per-pixel data-dependent narrowband weights)."""
    s, h, o = _clip01(sii), _clip01(ha), _clip01(oiii)
    wR = o ** (1.0 - o)
    R = wR * s + (1.0 - wR) * h
    m = o * h
    wG = m ** (1.0 - m)
    G = wG * h + (1.0 - wG) * o
    B = o
    return _clip01(R.bandjoin([G, B]))


def combine_star_rgb_vips(r: pyvips.Image, g: pyvips.Image, b: pyvips.Image) -> pyvips.Image:
    """Vips port of `combine_star_rgb`: stack three mono star channels to RGB."""
    return _clip01(r.bandjoin([g, b]))


def screen_vips(base: pyvips.Image, overlay: pyvips.Image) -> pyvips.Image:
    """Vips port of `screen`: 1 − (1−base)·(1−overlay)."""
    return 1.0 - (1.0 - _clip01(base)) * (1.0 - _clip01(overlay))
