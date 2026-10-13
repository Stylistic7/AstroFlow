"""Per-stack gain on the star flat's pattern — shape from the stars, amplitude
from the stack's own sky.

The star flat measures WHERE the sensor's etalon cells are with thousands of
stars per cell, but it measures their contrast for star continuum, and the
extended light of a stack sees the same pattern at its own contrast — 1.2x to
1.7x on the Cygnus mosaic, varying by panel and filter (2026-09-12 sweeps,
memory `project_star_flat_undercorrects_by_scale`). Dividing by the flat as
built therefore leaves a coherent fraction of the etalon in every master.

So each stack fits ONE number: in sensor coordinates, sub by sub, the
least-squares slope of the sub's band-passed relative sky on the flat's
band-passed pattern, medianed over the subs. k = 1 + slope, and the flat
handed to WBPP is 1 + k * (F - 1). The fit lives on the SUBS, never on the
master: a meridian flip rotates the sensor pattern 180 degrees between runs
and registration blends the two orientations in the master, so only a
per-sub, pre-registration fit sees the pattern whole — and WBPP applies the
scaled flat per frame, before registration, exactly as it applies any flat.

The etalon band is 256-1024 native px: the cells are 64-400 px blobs and the
fit runs at 1/8 resolution, where a 1-2% pattern on a sky of thousands of ADU
is measured to a few percent of itself per sub. A pedestal (bias, dark
current) left in during the division both dilutes the slope and prints an
inverted copy of the pattern — slope = ((k - 1) sky - pedestal) / (sky +
pedestal) — so the dark master is subtracted first when the stack has one;
without one the fit reads low and says so in its sidecar (`dark: False`).

Which subs: up to GAIN_MAX_SUBS of them, evenly spaced through the stack's
sorted file list so every run (night) is represented, each fitted on its
own; k is the MEDIAN of the per-sub slopes, so a cloudy or moonlit sub is a
vote, not a bias. The per-sub slopes are kept in the sidecar: their scatter
is the night-to-night transparency signal.

`fit_gain` is pure numpy; `fit_flat_gain` (stacks/star_flat.py) is the
@node around it. Edits here do not re-key that node (helpers are outside
its source hash — see project_color_scrgb_and_helper_eviction); bump
`GAIN_METHOD` to force a rebuild.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

GAIN_METHOD = 4                # 4: per-kind gain limits (a night-sky flat wants k~5-6)
                               # WITHHOLDS the flat (k = 0) instead of applying it whole
GAIN_BAND_PX = (256, 1024)   # the etalon band, native px
GAIN_SCALE = 8               # the fit runs at 1/GAIN_SCALE resolution
GAIN_MAX_SUBS = 40           # subs sampled per stack (evenly through the list)
GAIN_RANGE = (0.5, 2.5)      # a believable etalon amplitude for a stack's own sky
# k is an AMPLITUDE: 0 applies no star flat at all, 1 applies it exactly as
# built. So "I don't believe the fit" must not resolve to 1 — that is the
# maximum, not a neutral default. What distinguishes the two failures is the
# CORRELATION, not k: a fit that could not be measured has |corr| near 0,
# while a fit that measured the flat and found it wrong has |corr| near 1.
#   |corr| >= GAIN_CORR_TRUSTED and k below range -> the flat's pattern is
#       weaker in this stack's sky than in the flat. Apply it AS BUILT (k = 1)
#       and raise a Concern. Withholding it (k = 0, 2026-09-22 .. 10-09) threw
#       away its vignetting and dust correction too, though the fit only
#       measures the 256-1024 px band: 54 masters went out with no flat at
#       all (26 of them B, whose star flat carries 0.1% rms in that band).
#       The goal of the flat is a result GraXpert can finish; a flat that is
#       too strong in that band leaves far less than no flat (Stylistic7,
#       2026-10-09: "always apply the star flat").
#   |corr| >= GAIN_CORR_TRUSTED and k above range -> the sky wants more than
#       the flat has. Clamp to the range and warn.
#   |corr| below it -> nothing was measured; leave the group's flat as built.
# 2026-09-22: 49 of 124 fits refused, median fitted k 0.19, median corr -0.56;
# the Triangulum L stack fitted k=0.10 at corr -0.99 over 40 subs and had the
# full flat applied, punching holes where the flat's own artifacts were.
GAIN_CORR_TRUSTED = 0.5
GAIN_UNUSUAL = (0.7, 2.0)    # outside this it is applied but worth a look
# Per-KIND limits. The ranges above were tuned for the STAR flat, whose
# pattern is measured from continuum and wants k ~ 1.3-1.5. A NIGHT-SKY flat
# (stacks/sky_flat.py) is normalised by its members' TOTAL sky while the
# etalon rides on the LINE sky, so it comes out diluted: three panels on
# 2026-09-25 fitted k = 5.27 (H), 6.01 and 6.38 (S), each with per-sub scatter
# under 1.2x. Under the star flat's (0.5, 2.5) every one of them was CLAMPED
# to 2.5 and the flat applied at less than half the amplitude the sky wanted.
# The kind rides in `gain_args` and therefore in the cache key.
GAIN_LIMITS = {
    "star": (GAIN_RANGE, GAIN_UNUSUAL),
    "sky":  ((0.5, 8.0), (2.0, 8.0)),
}


def block_mean(a: np.ndarray, k: int) -> np.ndarray:
    """Downscale by integer block averaging (crops to a multiple of k)."""
    if k <= 1:
        return np.asarray(a, np.float64)
    h, w = a.shape
    a = np.asarray(a[: h - h % k, : w - w % k], np.float64)
    return a.reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def bandpass(x: np.ndarray, lo_px: float, hi_px: float, scale: int) -> np.ndarray:
    """Difference of gaussians selecting `lo_px`..`hi_px` native px."""
    return (gaussian_filter(x, lo_px / scale / 2.355)
            - gaussian_filter(x, hi_px / scale / 2.355))


def sky_model(med: np.ndarray) -> np.ndarray:
    """A degree-2 polynomial fitted to the fainter 70% of `med`: the sky
    level and its gradient. A polynomial, not a wide blur — a blur of a
    10% gradient leaves reflect-mode bumps inside 3 sigma of every edge that
    then masquerade as etalon-band structure (a +0.3 bias in the fitted
    slope on a synthetic ramp)."""
    h, w = med.shape
    yy, xx = np.mgrid[:h, :w]
    x = xx / w - 0.5; y = yy / h - 0.5
    A = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], -1).reshape(-1, 6)
    v = med.ravel()
    sel = np.isfinite(v) & (v < np.nanpercentile(v[np.isfinite(v)], 70))
    coef = np.linalg.lstsq(A[sel], v[sel], rcond=None)[0]
    return (A @ coef).reshape(h, w)


def relative_sky(img: np.ndarray, scale: int) -> np.ndarray:
    """Star-suppressed image divided by its sky model, minus one: the sky's
    structure as a fraction of the local sky level."""
    fill = np.nan_to_num(np.asarray(img, np.float64), nan=float(np.nanmedian(img)))
    k = max(3, int(round(48 / scale)) | 1)
    med = median_filter(fill, size=k)
    model = sky_model(med)
    return med / np.where(model > 0, model, np.nan) - 1.0


def sky_mask(img_ds: np.ndarray, rel: np.ndarray, band=GAIN_BAND_PX, scale=GAIN_SCALE) -> np.ndarray:
    """Interior (two sigma of the band's widest blur in from every edge),
    finite, and the fainter 70% — the bright 30% is nebula or stars, whose
    structure is not the flat's and only adds scatter."""
    h, w = rel.shape
    edge = int(2 * band[1] / scale / 2.355)
    yy, xx = np.mgrid[:h, :w]
    ok = np.isfinite(rel) & (yy >= edge) & (yy < h - edge) & (xx >= edge) & (xx < w - edge)
    lvl = median_filter(np.nan_to_num(img_ds, nan=0.0), size=5)
    return ok & (lvl < np.nanpercentile(lvl[ok], 70))


def in_units_of(dark: np.ndarray, sub: np.ndarray) -> np.ndarray:
    """The dark in the sub's units. A WBPP master is a PixInsight float XISF
    normalised to [0, 1] (1.0 = 65535 ADU) while a raw sub is uint16 ADU;
    subtracting one from the other removed ~0.01 instead of ~500 ADU and,
    at gain 0 where the H sky is 100-700 ADU above the pedestal, left the
    pedestal dominating the fit (boot 82: every k read 0.2-0.4)."""
    dark = np.asarray(dark, np.float64)
    if np.nanmax(dark) <= 1.0 and np.nanmax(sub) > 1.0:
        return dark * 65535.0
    return dark


def sample_paths(paths, max_subs: int) -> list:
    """Up to `max_subs` paths, evenly spaced through the list so every run
    (night) in the stack is represented."""
    paths = list(paths)
    if len(paths) <= max_subs:
        return paths
    idx = np.linspace(0, len(paths) - 1, max_subs).round().astype(int)
    return [paths[i] for i in dict.fromkeys(idx)]


def fit_gain(sub_paths, flat: np.ndarray, dark: np.ndarray | None, *,
             band=GAIN_BAND_PX, scale=GAIN_SCALE, max_subs=GAIN_MAX_SUBS,
             gain_range=GAIN_RANGE, read=None) -> dict:
    """The gain the stack's sky wants on `flat`'s pattern.

    Returns {k, slopes, corrs, n_subs, n_used, band, scale}. `k` is 1 +
    median(slopes) when the fit is believable (`gain_range`, >= 3 subs), else
    1.0 with `reason`. `read(path) -> ndarray` defaults to `file_io.generic.
    read_image`; a test passes its own.
    """
    if read is None:
        from file_io.generic import read_image
        read = lambda path: read_image(Path(path))   # noqa: E731 — the reader dispatches on Path
    F = block_mean(flat, scale)
    F = F / float(np.nanmedian(F))
    D = block_mean(dark, scale) if dark is not None else None
    pattern = bandpass(F - 1.0, band[0], band[1], scale)
    slopes, corrs = [], []
    chosen = sample_paths(sub_paths, max_subs)
    for p in chosen:
        img = block_mean(read(p), scale)
        if D is not None:
            img = img - in_units_of(D, img)
        img = img / F
        rel = relative_sky(img, scale)
        ok = sky_mask(img, rel, band, scale)
        if ok.sum() < 1000:
            continue
        r = bandpass(rel, band[0], band[1], scale)
        x, y = pattern[ok], r[ok]
        if not np.isfinite(x).all() or np.std(x) == 0:
            continue
        slopes.append(float(np.polyfit(x, y, 1)[0]))
        corrs.append(float(np.corrcoef(x, y)[0, 1]))
    out = dict(band=list(band), scale=scale, n_subs=len(list(sub_paths)), n_used=len(slopes),
               dark=dark is not None, slopes=slopes, corrs=corrs, method=GAIN_METHOD)
    if len(slopes) < 3:
        out.update(k=1.0, reason=f"only {len(slopes)} subs fit")
        return out
    k = 1.0 + float(np.median(slopes))
    corr = float(np.median(corrs))
    if gain_range[0] <= k <= gain_range[1]:
        out.update(k=k)
        return out
    if abs(corr) < GAIN_CORR_TRUSTED:
        out.update(k=1.0, k_fit=k, reason=f"fitted k={k:.2f} outside {gain_range} "
                                          f"and the fit is weak (|corr|={abs(corr):.2f}); "
                                          f"the flat is applied as built")
        return out
    if k < gain_range[0]:
        out.update(k=1.0, k_fit=k,
                   reason=f"fitted k={k:.2f} at |corr|={abs(corr):.2f} — this stack's sky shows "
                          f"less of the flat's pattern than the flat has; applied as built")
        return out
    out.update(k=gain_range[1], k_fit=k,
               reason=f"fitted k={k:.2f} above {gain_range[1]} at |corr|={abs(corr):.2f}; clamped")
    return out


def apply_gain(flat: np.ndarray, k: float) -> np.ndarray:
    """1 + k * (F - 1) about F's median — the flat WBPP will divide by."""
    F = np.asarray(flat, np.float64)
    F = F / float(np.nanmedian(F))
    return (1.0 + k * (F - 1.0)).astype(np.float32)


GAIN_SECTION = "star_flat_gain"   # the master's sidecar section: each scaled flat's fit, by flat id


@dataclass(frozen=True)
class FlatGain:
    """The gain `k` one stack's sky wants on a group flat's pattern
    (`stacks.star_flat.fit_flat_gain`, cached), naming the flat it scales by
    its STBLHASH (`flat_id`), with the whole fit as JSON for the record. A
    dataclass with `content_hash` so `@node` stamps it with its call hash.
    Stackers receive it among their `*cal_masters` and write the scaled flat
    into their own scratch (`with_scaled_flats`)."""
    k: float
    flat_id: str
    fit_json: str
    content_hash: str = ""


def with_scaled_flats(cal_by_id: dict, gains, tmp, models=()) -> dict:
    """`cal_by_id` (STBLHASH -> master path) with each gained flat replaced by
    its copy scaled to this stack, 1 + k (F - 1), written into `tmp` — the
    stacker's scratch, deleted with it, so no scaled flat outlives the stack
    (2026-10-09: 705 orphaned scaled flats, 157 GB, in the processed folder).
    The group flat's keywords ride along: WBPP matches the master to the
    stack by STBLHASH and recognises a pre-built one only as 'Master Flat'.

    Each dust-mote `model` (`stacks/dust_epochs.py`) adds a variant per gained
    flat under `"<flat id>:<model hash>"`: the scaled flat times the model's
    mote map, for the runs of that epoch — a synthetic flat is mote-free, the
    lights are not."""
    from stacks.dust_epochs import render_motes
    from stacks.dust_runs import BLOCK
    from file_io.generic import read_header, read_image
    from stacks.starflat import write_master_flat
    out = dict(cal_by_id)
    for g in gains:
        src = Path(out[g.flat_id])
        hdr = read_header(src)
        keywords = {key: hdr[key] for key in ("FILTER", "INSTRUME", "FOCALLEN", "XBINNING", "YBINNING",
                                              "EXPOSURE", "EXPTIME", "SFLATSUB", "SFLATCEL") if key in hdr}
        keywords.update({"IMAGETYP": "Master Flat", "STBLHASH": g.flat_id, "SFLATK": round(g.k, 4)})
        dest = Path(tmp) / f"scaled_flat_{g.flat_id}.xisf"
        scaled = apply_gain(np.asarray(read_image(src)).squeeze(), g.k)
        write_master_flat(scaled, dest, keywords)
        out[g.flat_id] = dest
        for model in models:
            variant = Path(tmp) / f"scaled_flat_{g.flat_id}_{model.content_hash}.xisf"
            write_master_flat(scaled * render_motes(model, scaled.shape, BLOCK), variant,
                              {**keywords, "DUSTMDL": model.content_hash})
            out[f"{g.flat_id}:{model.content_hash}"] = variant
    return out
