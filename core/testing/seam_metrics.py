"""Merger-agnostic seam / flatness metrics for an assembled mosaic.

Two panels offset in RA join along a roughly vertical seam, so a column-wise
sky-background profile is the natural probe: a seamless mosaic is ~one value
everywhere (stars aside), and a seam shows up as a step between the panels.
Shared by the PhotometricMosaic apex test and the OSS `merge_images_oss` seam
test so both judge "is there a visible seam?" the same way.
"""
import numpy as np


def seam_column_profile(mosaic: np.ndarray) -> dict:
    """Per-column sky-background median across the assembled mosaic, plus its
    peak-to-peak spread relative to the sky noise. Star pixels (6σ above the
    global median) are excluded so the profile tracks the background."""
    valid = mosaic > 0
    med = float(np.median(mosaic[valid]))
    mad = float(np.median(np.abs(mosaic[valid] - med))) + 1e-12
    star = mosaic > med + 6.0 * 1.4826 * mad
    keep = valid & ~star

    width = mosaic.shape[1]
    prof = np.full(width, np.nan)
    for x in range(width):
        col = mosaic[:, x][keep[:, x]]
        if col.size > 50:
            prof[x] = np.median(col)

    good = np.where(np.isfinite(prof))[0]
    sky_pixels = mosaic[keep]
    sky_rms = float(np.std(sky_pixels[sky_pixels < np.percentile(sky_pixels, 75)]))
    # Peak-to-peak over the valid span, ignoring the outer 2% (edge-erosion ramps).
    inner = good[int(0.02 * good.size): max(1, int(0.98 * good.size))]
    pp = float(np.nanmax(prof[inner]) - np.nanmin(prof[inner])) if inner.size else float("nan")
    return {
        "profile": prof, "good_cols": good,
        "sky_median": med, "sky_rms": sky_rms,
        "seam_pp_adu": pp, "seam_over_rms": pp / max(sky_rms, 1e-12),
    }


def join_step(seam: dict) -> dict:
    """Local seam discontinuity across the panel join — the field-independent
    "is there a visible seam?" invariant. Smooth the column sky profile, then
    take the largest change over a ~50 px window: a real join is a sharp jump,
    whereas a smooth gradient changes only gradually and is NOT a seam. Excludes
    the outer 2% of columns (footprint-edge ramps live there, the join doesn't)."""
    prof = seam["profile"]
    good = seam["good_cols"]
    if good.size < 200:
        return {"step_adu": float("nan"), "step_over_rms": float("nan")}
    m = int(0.02 * good.size)
    inner = good[m: good.size - m] if good.size - 2 * m > 200 else good
    p = prof[inner]
    k = 25
    sm = np.convolve(p, np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    win = 25
    if sm.size <= 2 * win:
        return {"step_adu": float("nan"), "step_over_rms": float("nan")}
    step = float(np.max(np.abs(sm[2 * win:] - sm[:-2 * win])))
    return {"step_adu": step, "step_over_rms": step / max(seam["sky_rms"], 1e-12)}
