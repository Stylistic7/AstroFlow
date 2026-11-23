"""Merger-agnostic seam / flatness metrics for an assembled mosaic.

Two panels offset in RA join along a roughly vertical seam, so a column-wise
sky-background profile is the natural probe: a seamless mosaic is ~one value
everywhere (stars aside), and a seam shows up as a step between the panels.
The OSS `merge_images_oss` seam test judges "is there a visible seam?" with it.
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


def _mtf(m: float, x: np.ndarray) -> np.ndarray:
    """PixInsight midtones transfer function."""
    return ((m - 1.0) * x) / (((2.0 * m - 1.0) * x) - m)


def _autostretch(channel: np.ndarray) -> np.ndarray:
    """The canonical display autostretch (finish_seed.js math): black point
    at median - 2.8 sigma, white at p99, midtones solved so the median
    lands at 0.25. This is what makes a 2e-5 background imbalance visible
    — the metric must amplify exactly the way the viewer does."""
    valid = channel[channel > 0]
    med = float(np.median(valid))
    sigma = 1.4826 * float(np.median(np.abs(valid - med)))
    bp = med - 2.8 * sigma
    wp = float(np.percentile(valid, 99.0))
    x = np.clip((channel - bp) / max(wp - bp, 1e-12), 0.0, 1.0)
    xm = (med - bp) / max(wp - bp, 1e-12)
    t = 0.25
    m = xm * (1.0 - t) / max(xm - 2.0 * xm * t + t, 1e-12)
    return _mtf(m, x)


def truth_chroma_bias(channels: dict, truths: dict,
                      grid: tuple = (8, 8)) -> dict:
    """Cross-channel BIAS against ground truth, in display units — the
    noise-immune successor to `stretched_chroma_spread` for labs that own
    the truth. Per channel: fit the free gauge (a, b) to the truth, take
    per-region medians of the residual (medians over ~10^4-10^5 px crush
    noise ~100x, so this measures bias, not SNR — the flaw that made the
    baseline-referenced metric report negative penalties), scale to
    display units via the channel's autostretch slope at background, and
    report the worst cross-channel disagreement of the centred region
    maps."""
    names = list(channels)
    centred = {}
    for name in names:
        out = channels[name]
        tr = truths[name]
        h = min(out.shape[0], tr.shape[0])
        w = min(out.shape[1], tr.shape[1])
        o, t = out[:h, :w], tr[:h, :w]
        valid = (o > 0) & (t > 0)
        a, b = np.polyfit(t[valid].ravel(), o[valid].ravel(), 1)
        resid = np.where(valid, o - (a * t + b), np.nan)

        # Display-slope at background: finite difference of the CHANNEL'S
        # autostretch (params from the full channel, evaluated at median).
        vals = o[valid]
        med = float(np.median(vals))
        sigma = 1.4826 * (float(np.median(np.abs(vals - med))) + 1e-12)
        bp = max(0.0, med - 2.8 * sigma)
        wp = max(float(np.percentile(vals, 99.0)), bp + 1e-3)
        x0 = (med - bp) / max(wp - bp, 1e-6)
        t = 0.25
        mid = float(np.clip(x0 * (1.0 - t) /
                            max(x0 - 2.0 * x0 * t + t, 1e-12), 1e-4, 0.999))
        eps = sigma

        def _st(x):
            xc = np.clip((x - bp) / max(wp - bp, 1e-6), 0.0, 1.0)
            return _mtf(mid, np.asarray([xc]))[0]

        slope = float((_st(med + eps) - _st(med)) / eps)

        gh, gw = grid
        regions = np.full((gh, gw), np.nan)
        for i in range(gh):
            for j in range(gw):
                cell = resid[i * h // gh:(i + 1) * h // gh,
                             j * w // gw:(j + 1) * w // gw]
                cell = cell[np.isfinite(cell)]
                if cell.size > 500:
                    regions[i, j] = np.median(cell) * slope / max(a, 1e-9)
        centred[name] = regions - np.nanmedian(regions)
    stack = np.stack([centred[n] for n in names])
    spread_map = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)
    return {
        "chroma_bias": float(np.nanmax(spread_map)),
        "spread_map": spread_map,
        "per_channel_regions": centred,
    }


def stretched_chroma_spread(channels: dict, grid: tuple = (4, 4)) -> dict:
    """Cross-channel background disagreement AFTER the display stretch —
    the number that matches the eye's "every panel has a different colour"
    verdict. Each channel is autostretched independently (as the viewer
    does), the sky background median is taken per region of a coarse
    grid (star pixels 6-sigma-rejected, zeros excluded), each channel's
    region map is centred on its own global level, and the spread is the
    worst cross-channel disagreement of those centred maps. A neutral-grey
    sky scores ~0; per-panel chroma quilting scores its display-visible
    amplitude (~0.01 = 1% display units is at the edge of visibility)."""
    names = list(channels)
    first = channels[names[0]]
    gh, gw = grid
    h, w = first.shape
    centred = {}
    for name in names:
        stretched = _autostretch(channels[name])
        raw = channels[name]
        valid = raw > 0
        med = float(np.median(raw[valid]))
        mad = float(np.median(np.abs(raw[valid] - med))) + 1e-12
        keep = valid & (raw < med + 6.0 * 1.4826 * mad)
        regions = np.full((gh, gw), np.nan)
        for i in range(gh):
            for j in range(gw):
                cell = keep[i * h // gh:(i + 1) * h // gh,
                            j * w // gw:(j + 1) * w // gw]
                vals = stretched[i * h // gh:(i + 1) * h // gh,
                                 j * w // gw:(j + 1) * w // gw][cell]
                if vals.size > 200:
                    regions[i, j] = np.median(vals)
        centred[name] = regions - np.nanmedian(regions)
    stack = np.stack([centred[n] for n in names])
    spread_map = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)
    return {
        "chroma_spread": float(np.nanmax(spread_map)),
        "spread_map": spread_map,
        "per_channel_regions": centred,
    }
