"""Dust motes in a night-sky flat: find them, divide their shadows out.

A night-sky flat is shot once and reused across epochs; its own motes are
the dust of the night it was shot, and dividing a later light by it would
print them back as bright "inverse motes". So the flat carries the etalon
and the vignetting only, and each epoch's dust comes from the lights (the
mote map, applied to synthetic flats when a Run is calibrated). Real
panel/twilight flats are never touched: their dust is supposed to be there.

Works on the flat at its BLOCK resolution (what `combine_sky_flat` holds
before it upsamples): fitting on the upsampled lattice imprints the lattice.

Prior (Stylistic7, 2026-10-13: "at most 3-5 motes in any image"): a mote is a
dark, round, compact dip — a few of them, deeper than the map's noise can
fake. Everything else in a flat is smooth (vignetting, etalon) or is not
a dip. Measured 2026-10-13 on the RC71's OIII flat: two motes, 3.5 % and
1.4 %, left at < 0.06 % by this model with the flat's texture intact.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve

SIGNIFICANCE = 8.0            # matched-filter sigma to call a dip a mote
MAX_MOTES = 5
SHARP = dict(flatten_px=256.0, radii_px=(24.0, 40.0, 64.0, 100.0), min_depth=0.010)
SOFT = dict(flatten_px=1200.0, radii_px=(160.0, 250.0, 320.0), min_depth=0.005)


def _disk(r: float) -> np.ndarray:
    n = int(np.ceil(r)); yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    d = (xx ** 2 + yy ** 2 <= r ** 2).astype(np.float32)
    return d / d.sum()


def _detect(flat_b: np.ndarray, block: int, flatten_px: float, radii_px: tuple, min_depth: float) -> list[dict]:
    """Matched-filter detection of dark disks on the block image. The map is
    first divided by its own broad smooth (the etalon and vignetting are
    smooth; a mote is not); the response at each radius is the mean dip over
    the disk, whose noise is the per-pixel noise / sqrt(area)."""
    flat = flat_b / ndimage.gaussian_filter(flat_b, flatten_px / block)
    dip = (1.0 - flat).astype(np.float32)
    best_sig = np.full(dip.shape, -np.inf, np.float32); best_r = np.zeros(dip.shape, np.float32)
    for r_px in radii_px:
        r = r_px / block
        resp = fftconvolve(dip, _disk(r), mode="same")      # the soft kernels are 80 blocks wide: direct is 10x slower
        # significance against the response map's own spread: whatever broad
        # structure the flatten left raises it honestly, and a mote is an
        # outlier of that map, not of the per-pixel noise
        sig = resp / (1.4826 * float(np.median(np.abs(resp - np.median(resp)))) or 1e-6)
        # the whole compactness ring (2 r) must be inside the frame: at the
        # edge the broad smooth misreads the vignetting and prints dips
        e = int(2 * r) + 2
        sig[:e, :] = sig[-e:, :] = sig[:, :e] = sig[:, -e:] = -np.inf
        better = sig > best_sig
        best_sig[better] = sig[better]; best_r[better] = r_px
    # local maxima in a window of the SMALLEST kernel, so a stronger, broader
    # dip nearby (an etalon cell) cannot hide a mote; duplicates are merged below
    window = int(2 * min(radii_px) / block) + 1
    peaks = (best_sig == ndimage.maximum_filter(best_sig, size=window)) & (best_sig >= SIGNIFICANCE)
    yy, xx = np.ogrid[:dip.shape[0], :dip.shape[1]]
    out = []
    for y, x in sorted(zip(*np.nonzero(peaks)), key=lambda p: -best_sig[p]):
        r = float(best_r[y, x]) / block
        d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        inside = float(np.mean(dip[d <= r]))
        if inside < min_depth or any(np.hypot(m["x"] / block - x, m["y"] / block - y) < max(r, m["radius_px"] / block) for m in out):
            continue
        # Compact: find where the dip ENDS (its radial profile under a fifth
        # of the core; the matched kernel is often smaller than a soft-rimmed
        # mote), then require the dip to be gone a radius further out. An
        # etalon cell is not compact. Shape beyond that cannot be trusted at
        # block resolution — the combine's block median softens every rim to
        # ~3 blocks — and what separates a mote from the etalon in a real
        # flat is depth: the etalon's mote-scale content is ~0.1 %, a mote
        # worth removing >= 1 %.
        core = float(np.median(dip[d <= max(0.5 * r, 1.0)]))
        ext = next((q for q in np.arange(r, 6 * r, 0.5) if np.any((d > q) & (d <= q + 0.5))
                    and float(np.median(dip[(d > q) & (d <= q + 0.5)])) < 0.2 * core), 6 * r)
        # the whole ring must lie inside the frame: a vignetted corner's
        # profile "ends" where the frame does and would read as compact
        ring = dip[(d > 1.4 * ext) & (d <= 2.0 * ext)]
        if core <= 0 or ext >= 6 * r or min(x, y, dip.shape[1] - 1 - x, dip.shape[0] - 1 - y) < 2.0 * ext \
                or (ring.size and float(np.mean(ring)) > inside / 3):
            continue
        out.append({"x": float(x) * block, "y": float(y) * block, "radius_px": float(best_r[y, x]),
                    "depth": round(inside, 4), "sigma": round(float(best_sig[y, x]), 1)})
    return out


def find_motes(flat_b: np.ndarray, block: int) -> list[dict]:
    """Sharp motes (sensor-window dust) and large soft ones (filter dust),
    strongest first, at most MAX_MOTES; coordinates and radii in full-res px."""
    sharp = _detect(flat_b, block, **SHARP)[:MAX_MOTES]
    soft = [m for m in _detect(flat_b, block, **SOFT)
            if not any(abs(m["x"] - s["x"]) < 200 and abs(m["y"] - s["y"]) < 200 for s in sharp)]
    return sharp + soft[:MAX_MOTES - len(sharp)]


STEP = 0.5   # profile radius step, blocks


def _patch(flat_b: np.ndarray, x: float, y: float, r0: float):
    """The local patch about (x, y), its coordinate grids, its masked-smooth
    reference (the mote does not dent its own reference) and the dip."""
    R = int(4 * r0) + 6
    y0, y1 = max(0, int(y) - R), min(flat_b.shape[0], int(y) + R)
    x0, x1 = max(0, int(x) - R), min(flat_b.shape[1], int(x) + R)
    patch = np.nan_to_num(flat_b[y0:y1, x0:x1], nan=1.0).astype(np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    far = (np.sqrt((xx - x) ** 2 + (yy - y) ** 2) > 2.0 * r0).astype(np.float32)
    sig = max(2.0 * r0, 2.0)
    ref = ndimage.gaussian_filter(patch * far, sig) / np.maximum(ndimage.gaussian_filter(far, sig), 1e-6)
    return patch, (y0, y1, x0, x1), yy, xx, ref, 1.0 - patch / np.maximum(ref, 1e-6)


def radial_profile(flat_b: np.ndarray, x: float, y: float, r0: float, recentre: bool = True
                   ) -> tuple[np.ndarray, float, tuple[float, float]]:
    """One mote's shadow as a radial profile of the dip (azimuthal median per
    STEP blocks of radius: the noise averaged over a ring, the edge kept)
    about a centroid-refined centre — a grid position leaves a dipole ring —
    with the extent where the shadow ends. Block units."""
    patch, _, yy, xx, ref, dip = _patch(flat_b, x, y, r0)
    cx, cy = float(x), float(y)
    pos = np.clip(dip, 0, None)
    for _ in range(3 if recentre else 0):
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        m = pos * (d <= 1.5 * r0); tot = float(m.sum())
        if tot <= 0:
            break
        cx, cy = float((m * xx).sum() / tot), float((m * yy).sum() / tot)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    nb = int(d.max() / STEP) + 2
    rb = np.minimum((d / STEP).astype(int), nb - 1)
    prof = np.array([float(np.median(dip[rb == i])) if np.any(rb == i) else np.nan for i in range(nb)], np.float32)
    # a ring with no block in it (the centre sits between blocks) takes
    # its neighbours' value, never 0: the core would read half as deep
    ok = np.isfinite(prof)
    prof = np.interp(np.arange(nb), np.flatnonzero(ok), prof[ok]).astype(np.float32) if ok.any() else np.zeros(nb, np.float32)
    core = float(prof[: max(int(0.5 * r0 / STEP), 2)].mean())
    ext = 3.0 * r0
    for i in range(int(0.6 * r0 / STEP), nb - 4):
        if prof[i: i + 4].max() < 0.02 * core:
            ext = i * STEP
            break
    prof = np.clip(prof, 0, None); prof[int(ext / STEP):] = 0.0
    return prof, ext, (cx, cy)


def _shadow(flat_b: np.ndarray, x: float, y: float, r0: float) -> tuple[np.ndarray, tuple[int, int, int, int], float, tuple[float, float]]:
    """One mote's multiplicative shadow S (<= 1) on a local patch of the block
    image: the radial profile, tapered at its extent, then one gentle 2D
    pass on what the radial model missed. Returns S, the patch bounds, the
    shadow's extent and the centre, all in block units."""
    prof, ext, (cx, cy) = radial_profile(flat_b, x, y, r0)
    patch, bounds, yy, xx, ref, _ = _patch(flat_b, x, y, r0)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    nb = len(prof)
    rb = np.minimum((d / STEP).astype(int), nb - 1)
    taper = np.clip((ext + 0.3 * r0 - np.arange(nb) * STEP) / max(0.3 * r0, 1.0), 0, 1)
    s1 = 1.0 - (prof * taper)[rb]
    resid = ndimage.gaussian_filter(1.0 - (patch / s1) / np.maximum(ref, 1e-6), 1.0)
    resid *= np.clip((ext + 0.3 * r0 - d) / max(0.3 * r0, 1.0), 0, 1)
    return (s1 * (1.0 - resid)).astype(np.float32), bounds, ext, (cx, cy)


def divide_motes(flat_b: np.ndarray, motes: list[dict], block: int) -> np.ndarray:
    """`flat_b` with each mote's shadow divided out; the motes gain `extent_px`
    and a refined `x`, `y`. The pixels keep their own values (PRNU, the
    etalon, noise) — only the shadow goes."""
    out = flat_b.astype(np.float32).copy()
    for m in motes:
        s, (y0, y1, x0, x1), ext, (cx, cy) = _shadow(flat_b, m["x"] / block, m["y"] / block, max(m["radius_px"] / block, 2.0))
        out[y0:y1, x0:x1] = flat_b[y0:y1, x0:x1] / s
        m.update(x=round(cx * block, 1), y=round(cy * block, 1), extent_px=round(ext * block, 1))
    return out
