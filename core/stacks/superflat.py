"""Data-derived superflat (sky/fringe flat) for narrowband etaloning.

STATUS (2026-09-06): SUPERSEDED for the etalon by stacks/starflat.py (a Gaia
star flat), which beats it on every independent judge. Retained as the
sky-based fringe-frame builder because the method is standard (IRAF
mkfringe / Astro-WISE) and would work on an archive of EMPTY, off-target
dithered fields — which this archive does not have: every target is centred,
so bright galaxies/nebula cores pile up into a ~1900 px-radius, +2.6% central
dome that no capping or masking removes (the object mask is local-relative,
so anything larger than its 64 px window passes unmasked), and the
per-pixel star residue (~2%) exceeds the etalon it is meant to capture.
Applied per-sub in WBPP it made masters WORSE at every scale. Measure any
future build with the radial-vs-panel-flat profile, not a high-pass.

A continuum flat panel cannot reproduce the etalon (interference) pattern
a narrowband EMISSION LINE prints on the sensor — the pattern is
wavelength-specific and the panel illuminates the whole passband as a
continuum (see the 2026-09-06 investigation). The correct reference is
the science data itself: many non-cospatial, dithered frames at the
target's own emission wavelength, combined UNREGISTERED in detector
coordinates so the sky averages out and the sensor-fixed multiplicative
response (etalon + vignetting + dust) survives. Standard supersky/fringe
construction (IRAF mkfringe, Astro-WISE, GaBoDS).

Method (principled, per the literature):
  1. dark-subtract each frame (remove additive amp glow / hot pixels /
     pedestal so what survives is purely multiplicative),
  2. normalize each frame to a common level (per-frame median),
  3. combine in DETECTOR coordinates (never registered) with a
     sigma-clipped mean — clipping rejects stars/bright cores, which with
     enough dithered fields are per-pixel outliers,
  4. normalize the result to 1.0 → the superflat, applied by DIVISION.

Streaming: a 2-pass sigma-clipped mean with per-pixel accumulators,
reading ONE frame at a time. Memory is O(one image) regardless of frame
count — so the whole multi-thousand-frame Ha pool combines without ever
holding it in memory (pyvips bandrank was considered but needs all N
images open at once; this is strictly lighter).
"""
import datetime
import numpy as np
import structlog
from pathlib import Path

import astropy.io.fits as pf

log = structlog.get_logger()


def _read(path: str | Path) -> np.ndarray:
    # memmap=False so BZERO/BSCALE scale correctly (memmap raises on scaled
    # int FITS); one full frame at a time keeps memory bounded.
    with pf.open(str(path), memmap=False) as h:
        return h[0].data.astype(np.float32)


def balance_by_pointing(paths: list[str], cap: int = 25) -> list[str]:
    """Keep at most `cap` frames per distinct ~1deg pointing.

    A superflat needs each detector pixel to see DIVERSE sky across the
    frame set so real structure averages out. Left unbalanced, a heavily
    imaged bright field (e.g. M31, 292 frames) dominates the center and
    imprints its glow into the flat. Capping per pointing forces the
    detector center to sample many different fields (2026-09-06)."""
    import json
    from collections import defaultdict
    by_pt: dict[tuple, list[str]] = defaultdict(list)
    for p in paths:
        try:
            c = json.load(open(p + ".json"))["astroflow"]["plate_solution"]["data"]["center"]
            key = (round(c[0]), round(c[1]))
        except Exception:
            key = ("?", len(by_pt))
        by_pt[key].append(p)
    out = []
    for frames in by_pt.values():
        out.extend(frames[:cap])
    log.info("balanced by pointing", pointings=len(by_pt), kept=len(out), cap=cap)
    return out


def _object_mask(frame: np.ndarray, thresh: float = 1.15) -> np.ndarray:
    """Boolean mask of stars/bright object pixels: those exceeding the
    frame's own smoothed background by more than `thresh`. Removes point
    sources and bright nebula cores per frame before combination."""
    from scipy.ndimage import uniform_filter
    bg = uniform_filter(frame, 64)
    return frame > bg * thresh


def build_master_dark(dark_paths: list[str], limit: int = 40) -> np.ndarray:
    """Mean of up to `limit` darks — a light additive reference; precision
    matters little for a normalized superflat, so a plain mean suffices."""
    acc = None
    n = 0
    for p in dark_paths[:limit]:
        d = _read(p)
        acc = d if acc is None else acc + d
        n += 1
    log.info("master dark built", n=n)
    return acc / n


def _frame_norm(path: str, dark: np.ndarray) -> float:
    """Per-frame normalization factor: median of the dark-subtracted frame
    from a strided (downsampled) read — cheap, robust to the exact value."""
    d = _read(path) - dark
    s = d[::8, ::8]
    m = float(np.median(s))
    return m if m > 1e-6 else 1.0


def combine_superflat(
    light_paths: list[str],
    dark: np.ndarray,
    out_path: str | Path,
    sigma: float = 3.0,
    progress_every: int = 100,
) -> np.ndarray:
    """Streaming 2-pass sigma-clipped-mean superflat in detector coords.

    Pass 0: per-frame normalization factors (strided median).
    Pass 1: per-pixel mean & std over dark-subtracted, normalized frames.
    Pass 2: per-pixel mean of values within `sigma` of the pass-1 mean
            (rejecting stars/cores). Result normalized to median 1.0.
    """
    H, W = dark.shape
    norms = []
    for i, p in enumerate(light_paths):
        norms.append(_frame_norm(p, dark))
        if i % progress_every == 0:
            log.info("superflat pass0 norms", done=i, total=len(light_paths))

    # Pass 1: object-masked sum and count → per-pixel mean/std of the
    # background×response only (stars and bright cores excluded so they
    # don't bias the model).
    s1 = np.zeros((H, W), np.float64)
    s2 = np.zeros((H, W), np.float64)
    cnt = np.zeros((H, W), np.float64)
    for i, (p, nf) in enumerate(zip(light_paths, norms)):
        f = (_read(p) - dark) / nf
        keep = ~_object_mask(f)
        s1 += np.where(keep, f, 0.0)
        s2 += np.where(keep, f * f, 0.0)
        cnt += keep
        if i % progress_every == 0:
            log.info("superflat pass1 mean/std", done=i, total=len(light_paths))
    denom = np.maximum(cnt, 1)
    mean = s1 / denom
    var = np.maximum(s2 / denom - mean * mean, 0.0)
    std = np.sqrt(var)

    # Pass 2: object-masked, sigma-clipped mean.
    csum = np.zeros((H, W), np.float64)
    ccnt = np.zeros((H, W), np.float64)
    lo = mean - sigma * std
    hi = mean + sigma * std
    for i, (p, nf) in enumerate(zip(light_paths, norms)):
        f = (_read(p) - dark) / nf
        keep = ~_object_mask(f) & (f >= lo) & (f <= hi)
        csum += np.where(keep, f, 0.0)
        ccnt += keep
        if i % progress_every == 0:
            log.info("superflat pass2 clipped mean", done=i, total=len(light_paths))
    clipped = np.where(ccnt > 0, csum / np.maximum(ccnt, 1), mean)

    superflat = clipped / np.median(clipped)
    # Write BEFORE logging so a summary-line error can never discard a
    # multi-hour combine (it did once: frames=n NameError, 2026-09-06).
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pf.PrimaryHDU(superflat.astype(np.float32)).writeto(out, overwrite=True)
    log.info(
        "superflat done", frames=len(light_paths),
        p1=float(np.percentile(superflat, 1)),
        p99=float(np.percentile(superflat, 99)),
    )
    return superflat.astype(np.float32)
