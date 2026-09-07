"""Forced aperture photometry at catalogue positions — the PixInsight-free
replacement for SpectrophotometricFluxCalibration's per-star measurement.

With solve-field's WCS the catalogue stars land within ~0.7 px of their
true positions, so there is no blind detection and no catalogue matching:
refine the centroid in a small window, then measure. PI's per-star
`img_signal` (a PSF-fit signal above the local background) is reproduced
to 1.4% rms per star — below the 3.5% astrophysical scatter of the
star-flat solve — by a 5 px radius aperture over an annulus-median
background (2026-09-07, 1,140 stars). Positions follow PI's convention in
the output: 0-based pixel coordinates with pixel centres at +0.5.

Everything is vectorised over stars (stamp cubes), so a 62 MP frame with
2,000 stars costs a few hundred ms after the file read.
"""
from __future__ import annotations

import numpy as np

# Geometry scales with the frame's typical FWHM (a 350 mm frame has ~1.8 px,
# a 2060 mm frame 5-8 px): a first pass on the brightest stars measures it,
# the second pass uses stamps of ~5 FWHM so the annulus clears the wings.
STAMP_R_MIN, STAMP_R_MAX = 18, 45
APERTURE_FWHM_SLOPE, APERTURE_FWHM_OFFSET = 2.0, 1.5     # r = 2.0*FWHM + 1.5 px
APERTURE_R_MIN = 3.5
ANNULUS_GAP, ANNULUS_R_MIN = 2.5, 9.0                     # annulus inner = max(aperture + gap, 9 or 3 FWHM), outer = stamp radius
FWHM_WEIGHT_SIGMA = 1.0   # px; Gaussian weight of the moment estimator (scaled with FWHM in pass 2)
FWHM_SCALE = 1 / 1.148    # moment FWHM -> PI PSF-fit FWHM (2026-09-07: ratio 1.148, corr 0.95)
CENTROID_R = 3.0
# PI rejects a star whose peak exceeds 0.75 x the brightest UNSATURATED star's
# peak in the frame (a saturated star fails its PSF fit and does not count):
# 0.4531 on an H sub whose brightest clean star peaks at 0.60, ~0.75 on an L
# sub with stars just below full scale (its kept stars reach 0.748).
SATURATION_FRACTION = 0.75
CHUNK_PIXELS = 2000 * 37 * 37   # stars per pass x stamp pixels: bounds the float32 temporaries to ~150 MB


def _geometry(stamp_r: int, fwhm: float) -> dict:
    yy, xx = np.mgrid[-stamp_r:stamp_r + 1, -stamp_r:stamp_r + 1]
    rr = np.hypot(xx, yy)
    ann_min = max(ANNULUS_R_MIN, 3.0 * fwhm)
    return {"stamp_r": stamp_r, "xx": xx.astype(np.float32), "yy": yy.astype(np.float32), "rr": rr,
            "annulus0": (rr >= ann_min) & (rr <= stamp_r), "ann_min": ann_min,
            "ap_max": min(max(12.0, 3.0 * fwhm), stamp_r - ANNULUS_GAP - 2.0), "centroid_r": max(CENTROID_R, 1.5 * fwhm),
            "weight_sigma": max(FWHM_WEIGHT_SIGMA, fwhm / 2.0), "core_r2": max(16.0, (2.5 * fwhm) ** 2)}


def measure(image: np.ndarray, x: np.ndarray, y: np.ndarray, *, iterations: int = 2) -> dict:
    """Photometry of the stars at FITS 1-based positions (x, y) on `image`
    (float32, normalised to full scale = 1). Returns arrays: x, y (refined,
    PI convention), signal, fwhm, peak, background, snr, ok, and the frame's
    saturation threshold and typical FWHM. Stars are processed in chunks so
    memory stays bounded whatever the field density."""
    # probe passes: the frame's typical FWHM from a sample of stars, starting at
    # the default geometry and re-probing once at the implied one (an 8 px PSF
    # measured with 1.8 px assumptions reads low but finite). Clipped to a
    # physical range: the moment estimator returns nonsense on saturated or
    # blended stars, and a frame with a handful of usable stars must not
    # explode the geometry (2069 mm L frame, 48 stars: 27.6 px -> NaNs).
    sample = np.linspace(0, len(x) - 1, min(len(x), 400)).astype(int) if len(x) else np.zeros(0, int)
    fwhm_typ = 1.8
    for _ in range(2):
        geo = _geometry(int(np.clip(round(5.0 * fwhm_typ + 9), STAMP_R_MIN, STAMP_R_MAX)), fwhm_typ)
        probe = _measure_chunk(image, x[sample], y[sample], iterations, geo)
        good = probe["ok"] & (probe["snr"] > 20) & np.isfinite(probe["fwhm"]) & (probe["fwhm"] < 15.0) & (probe["peak"] < 0.9)
        if good.sum() < 5:
            break
        fwhm_typ = float(np.clip(np.median(probe["fwhm"][good]), 1.0, 12.0))
    stamp_r = int(np.clip(round(5.0 * fwhm_typ + 9), STAMP_R_MIN, STAMP_R_MAX))
    geo = _geometry(stamp_r, fwhm_typ)
    chunk = max(200, CHUNK_PIXELS // (2 * stamp_r + 1) ** 2)
    parts = [_measure_chunk(image, x[i:i + chunk], y[i:i + chunk], iterations, geo) for i in range(0, len(x), chunk)]
    out = {k: np.concatenate([p[k] for p in parts]) if parts else np.zeros(0) for k in
           ("x", "y", "signal", "fwhm", "peak", "background", "snr", "ok")}
    out["fwhm_typical"] = fwhm_typ
    # PI's rule needs the brightest unsaturated peak over the WHOLE frame
    unsat = out["peak"][out["peak"] < 0.999]
    sat_ref = float(unsat.max()) if len(unsat) else 1.0
    out["ok"] = out["ok"] & (out["peak"] < SATURATION_FRACTION * sat_ref)
    out["saturation"] = SATURATION_FRACTION * sat_ref
    return out


def _measure_chunk(image: np.ndarray, x: np.ndarray, y: np.ndarray, iterations: int, geo: dict) -> dict:
    H, W = image.shape
    STAMP_R, xx, yy, _rr, _annulus0 = geo["stamp_r"], geo["xx"], geo["yy"], geo["rr"], geo["annulus0"]
    ANNULUS_R_MIN_, APERTURE_R_MAX, CENTROID_R_, WSIG, CORE_R2 = geo["ann_min"], geo["ap_max"], geo["centroid_r"], geo["weight_sigma"], geo["core_r2"]
    # FITS 1-based centre -> array index of the containing pixel
    ix = np.floor(x - 0.5).astype(int); iy = np.floor(y - 0.5).astype(int)
    fx = (x - 0.5) - ix; fy = (y - 0.5) - iy       # sub-pixel offset from that pixel's centre
    n = len(x)
    stamps = np.zeros((n, 2 * STAMP_R + 1, 2 * STAMP_R + 1), np.float32)
    ok = (ix >= STAMP_R) & (ix < W - STAMP_R) & (iy >= STAMP_R) & (iy < H - STAMP_R)
    for i in np.flatnonzero(ok):
        stamps[i] = image[iy[i] - STAMP_R:iy[i] + STAMP_R + 1, ix[i] - STAMP_R:ix[i] + STAMP_R + 1]
    bg = np.median(stamps[:, _annulus0], axis=1)
    noise = 1.4826 * np.median(np.abs(stamps[:, _annulus0] - bg[:, None]), axis=1)
    sub = stamps - bg[:, None, None]
    # centroid refinement: intensity-weighted mean within CENTROID_R of the current estimate
    cx, cy = fx.astype(np.float32), fy.astype(np.float32)
    pos = np.maximum(sub, 0)
    for _ in range(iterations):
        r2 = (xx[None] - cx[:, None, None]) ** 2 + (yy[None] - cy[:, None, None]) ** 2
        wts = np.where(r2 <= CENTROID_R_ ** 2, pos, np.float32(0))
        s = wts.sum((1, 2)); good = s > 0
        cx = np.where(good, (wts * xx).sum((1, 2)) / np.where(good, s, 1), cx)
        cy = np.where(good, (wts * yy).sum((1, 2)) / np.where(good, s, 1), cy)
    r2 = (xx[None] - cx[:, None, None]) ** 2 + (yy[None] - cy[:, None, None]) ** 2
    # FWHM: Gaussian-weighted second moment of the positive core, deconvolved
    # from the weight (sigma_star^2 = 1 / (1/var - 1/sigma_w^2)); calibrated
    # against PI's PSF-fit FWHM on the same stars (see FWHM_SCALE).
    def _moment_fwhm(wsig, core_r2):
        wgt = np.where(r2 <= core_r2, pos, np.float32(0)) * np.exp(-r2 / np.float32(2 * wsig ** 2))
        s = np.maximum(wgt.sum((1, 2)), 1e-12)
        var = np.maximum((wgt * r2).sum((1, 2)) / s / 2, 1e-6)
        inv = 1.0 / var - 1.0 / wsig ** 2                    # deconvolve the weight; collapses for stars broader than it
        return np.where(inv > 0.05 / wsig ** 2, FWHM_SCALE * 2.3548 / np.sqrt(np.maximum(inv, 1e-12)), np.nan)
    fwhm = _moment_fwhm(WSIG, CORE_R2)
    broad = ~np.isfinite(fwhm) | (fwhm > 2.5 * WSIG)       # re-measure broad stars with a wide weight
    if broad.any():
        fwhm = np.where(broad, _moment_fwhm(4.0 * WSIG, max(CORE_R2, (10.0 * WSIG) ** 2)), fwhm)
    fwhm = np.where(np.isfinite(fwhm), fwhm, 2.0 * WSIG)
    # aperture scaled with the star's FWHM — the region PI's PSF-fit signal
    # effectively integrates (2026-09-07: r_eq median 4.0 px, ~2.0-2.45 x FWHM)
    rad = np.clip(APERTURE_FWHM_SLOPE * fwhm + APERTURE_FWHM_OFFSET, APERTURE_R_MIN, APERTURE_R_MAX)
    ap = r2 <= (rad ** 2)[:, None, None]
    # a wide aperture pushes the annulus outward (per star); re-estimate its background
    wide = rad + ANNULUS_GAP > ANNULUS_R_MIN_
    if wide.any():
        ann = (_rr[None] >= (rad[wide] + ANNULUS_GAP)[:, None, None]) & (_rr[None] <= STAMP_R)
        bgw = np.array([np.median(st[a]) if a.any() else np.nan for st, a in zip(stamps[wide], ann)])
        bgw = np.where(np.isfinite(bgw), bgw, bg[wide])
        bg[wide] = bgw; sub[wide] = stamps[wide] - bgw[:, None, None]
    signal = (sub * ap).sum((1, 2))
    peak = (stamps * (r2 <= 4.0)).max((1, 2))
    snr = signal / np.maximum(noise * np.sqrt(ap.sum((1, 2))), 1e-12)
    ok &= (signal > 0) & (np.hypot(cx - fx, cy - fy) <= CENTROID_R_)
    return {"x": ix + cx + 0.5, "y": iy + cy + 0.5,      # PI convention: 0-based, centre at +0.5
            "signal": signal, "fwhm": fwhm, "peak": peak, "background": bg, "snr": snr, "ok": ok}
