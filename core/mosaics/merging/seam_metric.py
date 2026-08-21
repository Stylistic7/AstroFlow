"""Seam metrics — artifact-only quilt measures over panel overlaps.

Footprint-median statistics confound the artifact with the scene: panels
cover different sky, so their medians legitimately differ. Overlaps don't —
both panels record the SAME sky — so any disagreement there is pipeline
artifact by construction. Two gauges, matching the two error classes:

  * additive seam step  — median(plus − minus) per overlap
    (exactly `_level_fit_rows`' `c`; reused, not re-derived);
  * multiplicative seam ratio — robust median ratio of signal-above-floor
    per overlap, on pixels bright enough to carry gain information.

Acceptance reading: the level-matched merge should drive |step| to ~0;
SPFC gain normalization should drive ratios to ~1. A pipeline that leaves
either signature has un-corrected gauge of that class.
"""

import numpy as np

from mosaics.merging.oss import _intersect, _level_fit_rows, _sub

# Gain carries no information in pure background: (a−floor)/(b−floor) on
# sky pixels is noise/noise. Ratio pixels must sit meaningfully above the
# local floor on BOTH sides.
_RATIO_MIN_ABOVE_FLOOR = 5e-4


def seam_ratios(bboxes, masks, crops) -> list[dict]:
    """Per-overlap multiplicative disagreement: `{plus, minus, ratio, n}`
    with `ratio = median(plus_signal / minus_signal)` over pixels where both
    sides exceed their own floor by `_RATIO_MIN_ABOVE_FLOOR`. Overlaps with
    too little bright signal are omitted (a sky-only overlap cannot measure
    gain)."""
    rows = []
    n = len(crops)
    floors = [float(np.percentile(c[m], 5)) if c is not None and m.any() else 0.0
              for c, m in zip(crops, masks)]
    for i in range(n):
        if bboxes[i] is None:
            continue
        for j in range(i):
            if bboxes[j] is None:
                continue
            region = _intersect(bboxes[i], bboxes[j])
            if region is None:
                continue
            both = _sub(masks[i], bboxes[i], region) & _sub(masks[j], bboxes[j], region)
            if not both.any():
                continue
            a = _sub(crops[j], bboxes[j], region)[both] - floors[j]   # plus
            b = _sub(crops[i], bboxes[i], region)[both] - floors[i]   # minus
            bright = (a > _RATIO_MIN_ABOVE_FLOOR) & (b > _RATIO_MIN_ABOVE_FLOOR)
            if bright.sum() < 256:
                continue
            rows.append({
                "plus": j, "minus": i,
                "ratio": float(np.median(a[bright] / b[bright])),
                "n": int(bright.sum()),
            })
    return rows


def seam_report(bboxes, masks, crops) -> dict:
    """Summary of both seam gauges for one channel's placed panels.

    `steps`/`ratios` carry the per-overlap detail; the scalars are what an
    A/B table quotes: `step_p95` (additive artifact, image units) and
    `ratio_spread_p95` (multiplicative artifact, |ratio − 1|)."""
    fits = _level_fit_rows(bboxes, masks, crops)
    ratios = seam_ratios(bboxes, masks, crops)
    steps = np.array([abs(f["c"]) for f in fits]) if fits else np.array([0.0])
    rdev = (np.array([abs(r["ratio"] - 1.0) for r in ratios])
            if ratios else np.array([0.0]))
    return {
        "n_overlaps": len(fits),
        "n_ratio_overlaps": len(ratios),
        "step_median": float(np.median(steps)),
        "step_p95": float(np.percentile(steps, 95)),
        "ratio_dev_median": float(np.median(rdev)),
        "ratio_spread_p95": float(np.percentile(rdev, 95)),
        "steps": fits,
        "ratios": ratios,
    }
