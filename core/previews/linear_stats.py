"""Per-band linear statistics of a pixel sample — the numbers the tile
server ships as channelStats and the MCP image tools report."""
import numpy as np

_EMPTY = {"median": 0.0, "mad": 0.0, "sigma": 1.0, "min": 0.0, "max": 1.0,
          "p1": 0.0, "p95": 1.0, "p99": 1.0, "p999": 1.0, "high": 1.0}


def band_stats(sample: np.ndarray) -> list[dict]:
    """One dict per band of an (N, bands) sample: median, MAD, robust sigma,
    min/max, p1/p95/p99/p99.9, the sky subset's median and sigma, and the
    legacy `high` anchor.

    Coverage comes from every band together (`isfinite & > 0` — exact zero
    is the uncovered-canvas sentinel). The sky subset is the covered pixels
    whose mean-over-bands luminance sits below p60: background, largely
    star-free, so seeding black/midtones from it pins each channel's
    displayed sky to the same grey (the uniform-magenta finding, 2026-09-03).
    Narrowband channels differ wildly in SNR, so everything is per band.
    """
    arr = np.asarray(sample, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    nb = arr.shape[1]
    coverage = np.all(np.isfinite(arr), axis=1) & np.all(arr > 0.0, axis=1)
    sky_mask = coverage.copy()
    if coverage.sum() > 100:
        lum = arr[coverage].mean(axis=1)
        cut = np.percentile(lum, 60)
        idx = np.where(coverage)[0]
        sky_mask = np.zeros_like(coverage)
        sky_mask[idx[lum < cut]] = True
    stats = []
    for b in range(nb):
        c = arr[:, b]
        good = np.isfinite(c) & (c > 0.0)
        c = c[good]
        if c.size == 0:
            stats.append(dict(_EMPTY))
            continue
        med = float(np.median(c))
        mad = float(np.median(np.abs(c - med)))
        sigma = 1.4826 * mad if mad > 0 else max(med * 1e-3, 1e-6)
        p1, p95, p99, p999 = (float(np.percentile(c, q)) for q in (1, 95, 99, 99.9))
        sky = arr[:, b][sky_mask & good] if (sky_mask & good).any() else c
        sky_med = float(np.median(sky))
        sky_mad = float(np.median(np.abs(sky - sky_med)))
        stats.append({
            "median": med, "mad": mad, "sigma": sigma,
            "min": float(c.min()), "max": float(c.max()),
            "p1": p1, "p95": p95, "p99": p99, "p999": p999,
            "skyMedian": sky_med,
            "skySigma": 1.4826 * sky_mad if sky_mad > 0 else sigma,
            # p99.9 floored at median + 60σ so the gap stays wide relative to
            # noise for a faint channel (the built-in GHS finish reads it).
            "high": max(p999, med + 60.0 * sigma),
        })
    return stats
