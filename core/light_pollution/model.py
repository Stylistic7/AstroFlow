"""One site's sky glow as a small model, and what it says about any frame.

**City glow.** Light pollution is light domes over the cities around a site:
brightness rises toward the horizon, differently in each direction. The
model is log10 brightness relative to the zenith = (X−1)·F1(az) + (X−1)²·F2(az),
X the airmass, F1 and F2 Fourier series in azimuth to HARMONICS — N_CITY
numbers, `theta`. It is fitted in PERCENT of each frame's sky level per degree,
from moonless dark frames, on each frame's tilt against every frame of its
field (`measure.StackTilts`: within its run, and its run within its stack),
at every altitude and azimuth the field was shot over a season. The field's
own sky-fixed gradient — nebula, and what is left of the stack's mean glow —
is one free parameter per stack, scaled per frame by transparency over sky
level, partialled out of the fit (`robust.project_out`). Within a run alone
(to 2026-11-05) a run's ten degrees of drift pinned the dome's curvature and
left its slope to the model's form: the fit's spread was as large as the map.

**Moonlight.** Scattered moonlight has a known shape (Krisciunas & Schaefer
1991: separation from the Moon, phase, the Moon's airmass, the line of
sight's) and unknown amplitude. Per filter two amplitudes in counts per
second, `city` and `moon`, are fitted in ABSOLUTE units on all dark frames,
moonlit included, under non-negativity. The same field Moon-up and Moon-down
measures the moonlight directly, and its amplitude varies with the night's
aerosols (3x between training sets, 2026-11-02): per filter and NIGHT with
moonlit frames, the night's own (city, moon) are fitted beside the pooled
pair, under a prior NIGHT_PRIOR_WIDTH of the pooled value wide — a night
with little leverage keeps the pooled values, a night with much has its own
(`NightAmplitudes`, the record of the model; each travels into an
integration under its own identity, so a night's refit re-keys that night's
stacks and nothing else).

**Two regimes from one theta.** With the Moon down a frame's surface is its
sky level × (P/⟨P⟩ − 1), P = 10^(basis·theta): the percent form absorbs the
night's brightness. With the Moon up, the level is no longer the glow, so the
surface is exposure × [city·(P/⟨P⟩ − 1) + moon·(ks − ⟨ks⟩)], the full
moonlight field per cell, with the night's amplitudes when it has them. A
filter with too few moonlit runs keeps moon = 0 and is simply not
Moon-corrected; the report says so.

**Identity.** `content_hash` is of theta and the pooled amplitudes alone (the
MoteModel rule): counts, fit quality and the nights are the record. A refit
keeps the previous model, counts refreshed, unless the gate (GATE_*) says the
sky changed — otherwise every night's runs would re-key every integration.
This fold is path-dependent by design.

Imports nothing from compute_graph or stacks: `compute_graph.objects` holds a
`SiteModels` and imports this module at import time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from frozendict import frozendict

from hashing import hash_list
from light_pollution.robust import irls, irls_bounded, project_out
from light_pollution.sites import Site, same_site
from light_pollution.time import night_of

HARMONICS = 3
N_CITY = 2 * (2 * HARMONICS + 1)
K_EXT = 0.15                      # magnitudes per airmass, the extinction K&S scatter against
MOON_UP_DEG = -5.0                # above this the Moon lights the sky; below, the moonless regime
MIN_RUNS = 20                     # moonless dark runs a shape needs
MIN_EXPLAINED = 0.4               # of the tilt changes within stacks
MIN_MOON_FRAMES = 150             # moonlit frames a filter's moon amplitude needs
NIGHT_PRIOR_WIDTH = 1.0           # a night's own amplitude is expected within this fraction of the pooled one
FRAME_GATE_PCT_PER_DEG = 0.2      # below this a frame's surface is noise: not subtracted
ROUND = 4
REFERENCE_ALT = np.arange(30.0, 86.0, 5.0)   # where frames are shot; below, the (X-1)^2 terms amplify noise
REFERENCE_AZ = np.arange(0.0, 360.0, 15.0)
GRID_ALT = np.arange(15.0, 91.0, 5.0)        # the panel's map
GRID_AZ = np.arange(0.0, 360.0, 10.0)
# When a refit counts as a DIFFERENT model — a new identity, every corrected master re-integrated,
# deliberately, once — rather than the same sky with a night's more runs (the star-flat gate,
# stacks/star_flat.py). Both must hold, over the reference grid with each direction weighted by the
# dark frames shot there (a change where nobody shoots reaches no master):
#   significant: the rms change in the predicted tilt is at least GATE_SIGNIFICANCE times the fit's
#     own uncertainty — the spread of BOOTSTRAPS refits on the stacks resampled with replacement;
#   material: that rms change is at least FRAME_GATE_PCT_PER_DEG — smaller than the smallest tilt
#     ever subtracted from a frame, it cannot change a master.
# A change confined to one direction also publishes: at least GATE_LOCAL_PCT_PER_DEG and
# GATE_LOCAL_SIGNIFICANCE times its uncertainty, in a direction with frames. An amplitude publishes
# at GATE_AMPLITUDE of itself and GATE_SIGNIFICANCE times its uncertainty; a night's, at
# GATE_AMPLITUDE of itself.
# Before (to 2026-11-05): an unweighted 0.15 %/° rms over the whole grid, set from a synthetic refit;
# three new runs moved the south at altitude 30 — five frames there — by 2 %/° and re-keyed 143 masters.
BOOTSTRAPS = 20
GATE_SIGNIFICANCE = 2.0
GATE_LOCAL_PCT_PER_DEG = 1.0
GATE_LOCAL_SIGNIFICANCE = 5.0
GATE_AMPLITUDE = 0.2
# A model is trusted only where the site's frames are: a Fourier dome does what it likes elsewhere
# (Oakland, 2026-11-02: 45 runs over three octants predicted -100 % glow to the south). Coverage is
# counted in cells of COVERAGE_ALT_STEP x COVERAGE_AZ_STEP degrees with at least COVERAGE_FRAMES dark frames.
COVERAGE_ALT_STEP = 15.0
COVERAGE_AZ_STEP = 45.0
COVERAGE_FRAMES = 5
MAX_DARKER_THAN_ZENITH = 0.1      # a shape darker than the zenith by more than this inside its coverage is unphysical
MAX_BRIGHTER_THAN_ZENITH = 100.0  # city domes reach ~30x the zenith near the horizon; beyond this the fit is degenerate
                                  # (Arizona, 2026-11-02: 33 runs in four cells fitted 10^192 x inside them)


def coverage_cell(alt_deg: float, az_deg: float) -> tuple[int, int]:
    return int(min(alt_deg, 89.9) // COVERAGE_ALT_STEP), int((az_deg % 360.0) // COVERAGE_AZ_STEP)


def coverage_of(runs) -> tuple[tuple[int, int], ...]:
    counts: dict[tuple[int, int], int] = {}
    for t in runs:
        for j in range(t.n_frames):
            if t.dark[j]:
                c = coverage_cell(t.alt[j], t.az[j])
                counts[c] = counts.get(c, 0) + 1
    return tuple(sorted(c for c, n in counts.items() if n >= COVERAGE_FRAMES))


def covered(coverage, alt_deg: float, az_deg: float) -> bool:
    return coverage_cell(alt_deg, az_deg) in coverage


def airmass(alt_deg):
    z = np.radians(90.0 - np.clip(alt_deg, 3.0, 90.0))
    return 1.0 / np.sqrt(1.0 - 0.96 * np.sin(z) ** 2)


def basis(alt_deg, az_deg) -> np.ndarray:
    """(..., N_CITY): the city shape's terms at these directions; 0 at the zenith."""
    x1 = airmass(np.asarray(alt_deg, float)) - 1.0
    a = np.radians(np.asarray(az_deg, float))
    four = [np.ones_like(a)] + [f(k * a) for k in range(1, HARMONICS + 1) for f in (np.cos, np.sin)]
    return np.stack([x1 * t for t in four] + [x1 ** 2 * t for t in four], -1)


def city_shape(theta, alt_deg, az_deg) -> np.ndarray:
    """P: brightness relative to the zenith."""
    return 10.0 ** (basis(alt_deg, az_deg) @ np.asarray(theta, float))


def _moon_raw(rho_deg, alt_deg, moon_alt_deg, phase_deg):
    i = 10.0 ** (-0.4 * (3.84 + 0.026 * abs(phase_deg) + 4e-9 * phase_deg ** 4))
    f = 10.0 ** 5.36 * (1.06 + np.cos(np.radians(rho_deg)) ** 2) + 10.0 ** (6.15 - rho_deg / 40.0)
    return f * i * 10.0 ** (-0.4 * K_EXT * airmass(moon_alt_deg)) * (1.0 - 10.0 ** (-0.4 * K_EXT * airmass(alt_deg)))


_MOON_REFERENCE = float(_moon_raw(90.0, 45.0, 45.0, 0.0))


def moonlight(rho_deg, alt_deg, moon_alt_deg, phase_deg) -> np.ndarray:
    """Scattered moonlight, K&S 1991, relative to ρ = 90°, Moon and target at 45°, full."""
    return _moon_raw(np.asarray(rho_deg, float), np.asarray(alt_deg, float), moon_alt_deg, phase_deg) / _MOON_REFERENCE


# ------------------------------------------------------------------ the model
@dataclass(frozen=True)
class NightAmplitudes:
    """One night's amplitudes for one filter at one site, counts per second:
    what a moonlit frame of that night is corrected with. Its identity is its
    numbers, so it keys the integrations of that night's stacks by itself."""
    site: str
    night: str
    filter: str
    city: float
    moon: float
    content_hash: str = ""

    @classmethod
    def of(cls, site: str, night: str, filt: str, city: float, moon: float) -> "NightAmplitudes":
        return cls(site, night, filt, city, moon, hash_list([site, night, filt, city, moon]))


@dataclass(frozen=True)
class SiteModel:
    site: Site
    theta: tuple[float, ...]
    amplitudes: tuple[tuple[str, float, float], ...]   # (filter, city counts/s, moon counts/s)
    n_runs: int = 0
    n_frames: int = 0
    explained: float = 0.0
    explained_moonlit: float = 0.0
    newest_run: str = ""
    coverage: tuple[tuple[int, int], ...] = ()   # (alt band, octant) cells the site's frames sampled: the record
    nights: tuple[tuple[str, str, float, float], ...] = ()   # (night, filter, city, moon): the record
    content_hash: str = ""

    @classmethod
    def of(cls, site: Site, theta, amplitudes, **record) -> "SiteModel":
        theta = tuple(round(float(t), ROUND) for t in theta)
        amplitudes = tuple(sorted((str(f), round(float(c), ROUND), round(float(m), ROUND)) for f, c, m in amplitudes))
        return cls(site, theta, amplitudes, content_hash=hash_list([theta, amplitudes]), **record)

    def amplitude(self, filt: str) -> tuple[float, float] | None:
        return next(((c, m) for f, c, m in self.amplitudes if f == filt), None)

    def night(self, filt: str, night: str) -> NightAmplitudes | None:
        return next((NightAmplitudes.of(self.site.key, n, f, c, m) for n, f, c, m in self.nights
                     if f == filt and n == night), None)

    def summary(self) -> dict:
        return {"hash": self.content_hash, "runs": self.n_runs, "frames": self.n_frames,
                "explained": round(self.explained, 3), "explained_moonlit": round(self.explained_moonlit, 3),
                "newest_run": self.newest_run, "coverage": [list(c) for c in self.coverage],
                "amplitudes": {f: {"city": c, "moon": m, "nights": sum(1 for _, nf, _, _ in self.nights if nf == f)}
                               for f, c, m in self.amplitudes}}


@dataclass(frozen=True)
class SiteModels:
    """Every site's model, by site key, and the record of every site measured
    (with a model or the reason it has none): the Observing Sites panel's data."""
    models: frozendict
    report: tuple
    content_hash: str = ""

    @classmethod
    def empty(cls) -> "SiteModels":
        return cls(frozendict(), (), "none")

    @classmethod
    def of(cls, models: dict, report: tuple) -> "SiteModels":
        identity = hash_list([(k, models[k].content_hash) for k in sorted(models)]) if models else "none"
        return cls(frozendict(models), tuple(report), identity)

    def model_for(self, position: tuple[float, float] | None) -> SiteModel | None:
        if position is None:
            return None
        return next((m for m in self.models.values() if same_site((m.site.lat, m.site.lon), position)), None)


# ------------------------------------------------------------------ fit
# A frame against its whole field: its tilt within its run plus its run's tilt `g` within the stack,
# the latter scaled by the frame's transparency `scale` (the run's median is scale x the frame's sky).
# What the stack's frames then share is the field's sky-fixed gradient — the nuisance — in counts per
# degree, which reaches a frame's row scaled by `scale` x the run's scale `sc` against the stack and
# divided by what the row is in units of: the frame's sky level (percent) or its exposure (counts/s).
def _y_pct(t, j, g, sc):
    return np.asarray(t.tilt_pct[j]) + 100.0 * t.scale[j] * np.asarray(g) / t.level[j]


def _x_pct(t, j):
    return t.city_pct[j]


def _w_pct(t, j, sc):
    return 100.0 * t.scale[j] * sc / t.level[j]


def _y_abs(t, j, g, sc):
    return np.asarray(t.tilt_abs[j]) + t.scale[j] * np.asarray(g) / t.exposure[j]


def _w_abs(t, j, sc):
    return t.scale[j] * sc / t.exposure[j]


def _rows(stacks, rows_of, y_of, x_of, w_of) -> list:
    """Per stack with at least two chosen frames: (y, X, w, frames, runs, filter) — each frame's two
    tilt components as two rows of y and X, w the rows' coefficient on the stack's nuisance."""
    blocks = []
    for s in stacks:
        y, X, w, runs = [], [], [], 0
        for t, g, sc in zip(s.runs, s.run_tilt, s.run_scale):
            idx = rows_of(t)
            runs += bool(idx)
            for j in idx:
                yv, wv = y_of(t, j, g, sc), w_of(t, j, sc)
                if np.all(np.isfinite(yv)) and np.isfinite(wv):
                    y.append(yv)
                    X.append(x_of(t, j))
                    w.append(wv)
        if len(y) < 2:
            continue
        blocks.append((np.array(y, float).reshape(-1), np.array(X, float).reshape(2 * len(y), -1),
                       np.repeat(np.array(w, float), 2), len(y), runs, s.runs[0].filter))
    return blocks


def _stacked(blocks) -> tuple[np.ndarray, np.ndarray, tuple]:
    """(Y, X, nuisance) of these stacks' blocks, stacked; nuisance = (w, groups) as the robust fits
    take it — per stack two groups, its x-tilt rows and its y-tilt rows: the field's gradient is a
    2-vector, one free coefficient per component."""
    if not blocks:
        return np.zeros(0), np.zeros((0, 1)), (np.zeros(0), [])
    Y, X, W = (np.concatenate([b[i] for b in blocks]) for i in range(3))
    ends = np.cumsum([len(b[0]) for b in blocks])
    groups = [np.arange(e - len(b[0]) + c, e, 2) for b, e in zip(blocks, ends) for c in (0, 1)]
    return Y, X, (W, groups)


def _bootstrap(blocks, fit, rng) -> np.ndarray:
    """(BOOTSTRAPS, k): `fit`'s coefficients on the stacks resampled with replacement — the fit's own uncertainty."""
    out = []
    for _ in range(BOOTSTRAPS):
        Y, X, nu = _stacked([blocks[k] for k in rng.integers(0, len(blocks), len(blocks))])
        out.append(fit(X, Y, nuisance=nu)[0])
    return np.array(out)


def _share(coef, X, Y, inl) -> float:
    """The share of the (robust) variance the model explains: MAD-based, so a
    cloud-crossed run cannot make a good model look useless."""
    if inl.sum() < 4:
        return 0.0
    y, r = Y[inl], Y[inl] - X[inl] @ coef
    spread = 1.4826 * np.median(np.abs(y - np.median(y)))
    if spread == 0:
        return 0.0
    return float(1 - (1.4826 * np.median(np.abs(r - np.median(r))) / spread) ** 2)


def _explained(coef, X, Y, inl, nuisance, by=None) -> float | dict:
    """`_share` with the nuisance partialled out at the inliers; with `by`, a label per row, per label."""
    if not len(Y):
        return {} if by is not None else 0.0
    X, Y = project_out(X, Y, *nuisance, inl.astype(float))
    if by is None:
        return _share(coef, X, Y, inl)
    return {k: round(_share(coef, X[by == k], Y[by == k], inl[by == k]), 3) for k in sorted(set(by))}


def _tilt_grid(theta) -> np.ndarray:
    """The shape's gradient along altitude, %/°, over the reference grid — what two models are
    compared by. `theta` (N_CITY,) → (alt, az); (n, N_CITY) → (alt, az, n)."""
    alt, az = np.meshgrid(REFERENCE_ALT, REFERENCE_AZ, indexing="ij")
    return 100.0 * math.log(10.0) * ((basis(alt + 0.5, az) - basis(alt - 0.5, az)) @ np.asarray(theta, float).T)


def _frame_weights(runs) -> np.ndarray:
    """Dark frames per reference-grid direction: where a change in the shape can reach a master."""
    w = np.zeros((len(REFERENCE_ALT), len(REFERENCE_AZ)))
    d_alt, d_az = REFERENCE_ALT[1] - REFERENCE_ALT[0], REFERENCE_AZ[1] - REFERENCE_AZ[0]
    for t in runs:
        for j in range(t.n_frames):
            i = round((t.alt[j] - REFERENCE_ALT[0]) / d_alt)
            if t.dark[j] and 0 <= i < len(REFERENCE_ALT):
                w[i, round(t.az[j] / d_az) % len(REFERENCE_AZ)] += 1
    return w


def _gate(new: SiteModel, previous: SiteModel | None, weights: np.ndarray, tilt_sigma: np.ndarray,
          amp_sigma: dict) -> dict:
    """Whether `new` replaces `previous`, and the numbers that decided it (the report's `gate`).
    `tilt_sigma` is one fit's uncertainty per reference direction; two fits of nearly the same
    stacks differ by sqrt(2) of it. `amp_sigma`: per filter, (city, moon) uncertainties."""
    if not weights.any():
        weights = np.ones_like(weights)
    w = weights / weights.sum()
    noise = math.sqrt(2.0) * tilt_sigma
    noise_rms = float(np.sqrt((w * noise ** 2).sum()))
    if previous is None:
        return {"previous": None, "kept": False, "noise": round(noise_rms, 3)}
    d = np.abs(_tilt_grid(new.theta) - _tilt_grid(previous.theta))
    change = float(np.sqrt((w * d ** 2).sum()))
    sampled = weights > 0
    local = float(d[sampled].max())
    local_sig = float((d / np.maximum(noise, 1e-9))[sampled].max())
    shape = ((change >= GATE_SIGNIFICANCE * noise_rms and change >= FRAME_GATE_PCT_PER_DEG)
             or (local >= GATE_LOCAL_PCT_PER_DEG and local_sig >= GATE_LOCAL_SIGNIFICANCE))
    old = {f: (c, m) for f, c, m in previous.amplitudes}
    amps = set(old) != {f for f, _, _ in new.amplitudes}
    for f, c, m in new.amplitudes:
        for a, b, s in zip((c, m), old.get(f, (0.0, 0.0)), amp_sigma.get(f, (0.0, 0.0))):
            if (abs(a - b) > 1e-3 and abs(a - b) >= GATE_AMPLITUDE * max(abs(a), abs(b))
                    and abs(a - b) >= GATE_SIGNIFICANCE * math.sqrt(2.0) * s):
                amps = True
    return {"previous": previous.content_hash, "kept": not (shape or amps), "change": round(change, 3),
            "noise": round(noise_rms, 3), "local": round(local, 2), "local_significance": round(local_sig, 1),
            "amplitudes_changed": amps}


def _night_amplitudes(site: Site, stacks, model: SiteModel, scale: dict, dark, moonless) -> dict:
    """Per filter with a moon amplitude, each night with moonlit frames: its own (city, moon) from all
    its dark frames — one city column and one moon column per night — each under a Gaussian prior at
    the pooled value, NIGHT_PRIOR_WIDTH of it wide: one row per column with that pull at the pooled
    fit's residual `scale`, which also judges the frames' Tukey weights, so a night that differs
    from the pooled sky by more than the noise keeps its frames and loses the prior, not the reverse."""
    theta = np.asarray(model.theta)
    out = {}
    for filt, city, moon in model.amplitudes:
        if moon == 0.0 or filt not in scale:
            continue
        of_filter = [s for s in stacks if s.runs[0].filter == filt]
        night = {t.run: night_of(t.date, site.lon) for s in of_filter for t in s.runs}
        nights = sorted({night[t.run] for s in of_filter for t in s.runs if dark(t)})
        moon_nights = sorted({night[t.run] for s in of_filter for t in s.runs if len(dark(t)) > len(moonless(t))})
        col = {n: i for i, n in enumerate(nights)}
        mcol = {n: len(nights) + i for i, n in enumerate(moon_nights)}

        def x_of(t, j):
            X = np.zeros((2, len(nights) + len(moon_nights)))
            X[:, col[night[t.run]]] = np.asarray(t.city_pct[j]) @ theta / 100.0
            if night[t.run] in mcol:
                X[:, mcol[night[t.run]]] = t.moon[j]
            return X

        Y, X, (W, groups) = _stacked(_rows(of_filter, dark, _y_abs, x_of, _w_abs))
        if not len(Y):
            continue
        pooled = np.repeat([city, moon], [len(nights), len(moon_nights)])
        pull = scale[filt] / (NIGHT_PRIOR_WIDTH * np.maximum(pooled, 1e-3))
        coef, _, _ = irls_bounded(np.vstack([X, np.diag(pull)]), np.concatenate([Y, pull * pooled]),
                                  nuisance=(np.concatenate([W, np.zeros(len(pull))]), groups),
                                  prior=len(pull), scale=scale[filt])
        for n in moon_nights:
            out[n, filt] = (float(coef[col[n]]), float(coef[mcol[n]]))
    return out


def _kept_nights(new: dict, previous) -> tuple:
    """(night, filter, city, moon) per night fitted; the previous values stay where neither moved by
    GATE_AMPLITUDE of itself — a night's stacks re-integrate for the night's own change, not for drift."""
    old = {(n, f): (c, m) for n, f, c, m in previous}
    out = []
    for (n, f), (c, m) in sorted(new.items()):
        if (n, f) in old and all(abs(a - b) < GATE_AMPLITUDE * max(abs(a), abs(b), 1e-9)
                                 for a, b in zip((c, m), old[n, f])):
            c, m = old[n, f]
        out.append((n, f, round(c, ROUND), round(m, ROUND)))
    return tuple(out)


def sigma_grid(thetas: np.ndarray) -> list:
    """The bootstrap spread of `brightness_grid`'s percent, rows by altitude, columns by azimuth —
    how sure the map is in each direction. None where a refit ran away."""
    alt, az = np.meshgrid(GRID_ALT, GRID_AZ, indexing="ij")
    with np.errstate(over="ignore", invalid="ignore"):
        s = (100.0 * city_shape(thetas.T, alt, az)).std(-1, ddof=1).round(1)
    return [[float(v) if np.isfinite(v) else None for v in row] for row in s]


MAX_SAMPLES = 2000


def _samples(runs) -> list:
    """(alt, az, moonless) of the dark frames, at most MAX_SAMPLES — the panel's dots."""
    pts = [(round(t.alt[j], 1), round(t.az[j], 1), bool(t.moonless[j]))
           for t in runs for j in range(t.n_frames) if t.dark[j]]
    return pts[::max(1, -(-len(pts) // MAX_SAMPLES))]


def fit_site_model(site: Site, stacks, previous: SiteModel | None) -> tuple[SiteModel | None, dict]:
    """The site's model from its stacks' tilts (`light_pollution.measure.StackTilts`),
    and the site's report; (None, report) when the data do not support a model."""
    stacks = [s for s in stacks if s is not None and s.runs]
    runs = [t for s in stacks for t in s.runs]
    dark = lambda t: [j for j in range(t.n_frames) if t.dark[j]]                          # noqa: E731
    moonless = lambda t: [j for j in range(t.n_frames) if t.dark[j] and t.moonless[j]]  # noqa: E731
    report = {"key": site.key, "lat": site.lat, "lon": site.lon,
              "runs": len(runs), "stacks": len(stacks), "frames": sum(t.n_frames for t in runs),
              "usable_frames": sum(len(dark(t)) for t in runs),
              "moonlit_frames": sum(len(dark(t)) - len(moonless(t)) for t in runs),
              "newest_run": max((t.date for t in runs), default=""),
              "samples": _samples(runs), "model": None, "reason": ""}
    rng = np.random.default_rng(0)                   # per call: a refit's bootstrap is bit-identical
    # 1. the shape, in percent of the sky level, from moonless frames
    blocks = _rows(stacks, moonless, _y_pct, _x_pct, _w_pct)
    n_runs, n_frames = sum(b[4] for b in blocks), sum(b[3] for b in blocks)
    if n_runs < MIN_RUNS:
        report["reason"] = f"needs at least {MIN_RUNS} moonless dark runs; has {n_runs}"
        return None, report
    Y, X, nu = _stacked(blocks)
    theta, inl, _ = irls(X, Y, nuisance=nu)
    explained = _explained(theta, X, Y, inl, nu)
    report["explained_by_filter"] = _explained(theta, X, Y, inl, nu, np.concatenate([[b[5]] * len(b[0]) for b in blocks]))
    if explained < MIN_EXPLAINED:
        report["reason"] = f"the shape explains only {100 * explained:.0f}% of the tilt changes within stacks"
        return None, report
    coverage = coverage_of(runs)
    shape = [float(city_shape(theta, (a + 0.5) * COVERAGE_ALT_STEP, (z + 0.5) * COVERAGE_AZ_STEP)) for a, z in coverage]
    darkest, brightest = (min(shape), max(shape)) if shape else (1.0, 1.0)
    if darkest < 1.0 - MAX_DARKER_THAN_ZENITH:
        report["reason"] = (f"the shape is {100 * (1 - darkest):.0f}% darker than the zenith inside the site's own "
                            f"coverage: unphysical, the stacks do not pin it down")
        return None, report
    if not brightest <= MAX_BRIGHTER_THAN_ZENITH:                    # also catches inf and nan
        report["reason"] = (f"the shape is {brightest:.3g}x brighter than the zenith inside the site's own "
                            f"coverage: a degenerate fit, not a sky")
        return None, report
    thetas = _bootstrap(blocks, irls, rng)
    # 2. the pooled amplitudes, in counts per second, per filter, from all dark frames
    amplitudes, amp_sigma, amp_scale = [], {}, {}
    for filt in sorted({t.filter for t in runs}):
        of_filter = [s for s in stacks if s.runs[0].filter == filt]
        moonlit = sum(len(dark(t)) - len(moonless(t)) for s in of_filter for t in s.runs)
        with_moon = moonlit >= MIN_MOON_FRAMES
        blocks_f = _rows(
            of_filter, dark, _y_abs,
            lambda t, j: np.column_stack([np.asarray(t.city_pct[j]) @ theta / 100.0,
                                          np.asarray(t.moon[j])])[:, :2 if with_moon else 1], _w_abs)
        if len(blocks_f) < 3:
            continue
        Ya, Xa, nua = _stacked(blocks_f)
        coef, _, amp_scale[filt] = irls_bounded(Xa, Ya, nuisance=nua)
        amplitudes.append((filt, float(coef[0]), float(coef[1]) if with_moon else 0.0))
        sigma = _bootstrap(blocks_f, irls_bounded, rng).std(0, ddof=1)
        amp_sigma[filt] = (float(sigma[0]), float(sigma[1]) if with_moon else 0.0)
    # 3. how well the whole model does on moonlit frames
    amp = {f: (c, m) for f, c, m in amplitudes}
    Ym, Xm, num = _stacked(_rows(
        [s for s in stacks if s.runs[0].filter in amp], lambda t: [j for j in dark(t) if not t.moonless[j]],
        _y_abs,
        lambda t, j: amp[t.filter][0] * (np.asarray(t.city_pct[j]) @ theta / 100.0)[:, None]
        + amp[t.filter][1] * np.asarray(t.moon[j])[:, None], _w_abs))
    explained_moonlit = _explained(np.ones(1), Xm, Ym, np.ones(len(Ym), bool), num)
    record = dict(n_runs=n_runs, n_frames=n_frames, explained=explained,
                  explained_moonlit=explained_moonlit, newest_run=report["newest_run"], coverage=coverage)
    model = SiteModel.of(site, theta, amplitudes, **record)
    # 4. the same sky, or a different one: the gate, against the fit's own uncertainty where frames are
    gate = _gate(model, previous, _frame_weights(runs), _tilt_grid(thetas).std(-1, ddof=1), amp_sigma)
    if gate["kept"]:
        model = replace(previous, **record)
    # 5. each night's own amplitudes, under the model that stands
    nights = _kept_nights(_night_amplitudes(site, stacks, model, amp_scale, dark, moonless),
                          previous.nights if gate["kept"] else ())
    gate["nights"] = len(nights)
    model = replace(model, nights=nights)
    report.update(model=model.summary(), gate=gate, sigma=sigma_grid(thetas))
    return model, report


# ------------------------------------------------------------------ a frame's surface
def frame_surface(model: SiteModel, header: dict, wcs_header: dict, shape: tuple[int, int],
                  level: float, filt: str, night: NightAmplitudes | None = None, step: int = 64):
    """The glow in this frame's own pixel grid at its mid-exposure moment, as
    (surface in counts with mean 0, or None; its tilt in % of `level` per
    degree; the regime: 'moonless', 'moonlit' (with `night`'s amplitudes),
    'moonlit-pooled', 'gated', 'no-amplitude' or 'uncovered' — outside the
    directions the site's frames sampled). `level` is the frame's sky level
    in counts (lower-half median of its block medians); `header` carries
    DATE-OBS, EXPTIME, SITELAT/SITELONG."""
    from astropy import units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, get_sun
    from astropy.io import fits
    from astropy.time import Time
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales
    from scipy.ndimage import zoom

    from light_pollution.sites import site_of
    from light_pollution.time import mid_exposure

    ny, nx = shape
    ys, xs = np.arange(step / 2 - 0.5, ny, step), np.arange(step / 2 - 0.5, nx, step)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    wcs = WCS(fits.Header({k: v for k, v in wcs_header.items()
                           if isinstance(v, (int, float, str)) and k not in ("COMMENT", "HISTORY")}))
    ra, dec = wcs.all_pix2world(gx.ravel(), gy.ravel(), 0)
    lat, lon = site_of(header)
    t = Time(mid_exposure(header), scale="utc")
    frame = AltAz(obstime=t, location=EarthLocation(lat=lat * u.deg, lon=lon * u.deg))
    cells = SkyCoord(ra * u.deg, dec * u.deg)
    aa = cells.transform_to(frame)
    centre_az = float(np.degrees(math.atan2(np.median(np.sin(aa.az.rad)), np.median(np.cos(aa.az.rad)))) % 360)
    if model.coverage and not covered(model.coverage, float(np.median(aa.alt.deg)), centre_az):
        return None, 0.0, "uncovered"
    p = city_shape(model.theta, aa.alt.deg, aa.az.deg)
    city = p / p.mean() - 1.0
    moon = get_body("moon", t, frame.location)
    moon_alt = float(moon.transform_to(frame).alt.deg)
    regime = "moonless"
    if moon_alt > MOON_UP_DEG:
        amp = (night.city, night.moon) if night is not None else model.amplitude(filt)
        if amp is None:
            return None, 0.0, "no-amplitude"
        phase = 180.0 - float(moon.separation(get_sun(t)).deg)
        ks = moonlight(cells.separation(moon).deg, aa.alt.deg, moon_alt, phase)
        surface = float(header["EXPTIME"]) * (amp[0] * city + amp[1] * (ks - ks.mean()))
        regime = "moonlit" if night is not None else "moonlit-pooled"
    else:
        surface = level * city
    surface = surface.reshape(gy.shape)
    scale = float(proj_plane_pixel_scales(wcs)[0])          # degrees per pixel
    a = np.stack([np.ones(surface.size), (gx.ravel() - gx.mean()) * scale, (gy.ravel() - gy.mean()) * scale], 1)
    g = np.linalg.lstsq(a, surface.ravel(), rcond=None)[0][1:]
    tilt = float(100.0 * np.hypot(*g) / level) if level > 0 else 0.0
    if tilt < FRAME_GATE_PCT_PER_DEG:
        return None, tilt, "gated"
    full = zoom(surface, (ny / surface.shape[0], nx / surface.shape[1]), order=1)[:ny, :nx]
    return full.astype(np.float32), tilt, regime


def brightness_grid(model: SiteModel) -> list:
    """Percent brighter than the zenith over GRID_ALT x GRID_AZ — the panel's map.
    None outside the model's coverage, where the shape is not trusted (`frame_surface` corrects nothing there)."""
    alt, az = np.meshgrid(GRID_ALT, GRID_AZ, indexing="ij")
    pct = (100.0 * (city_shape(model.theta, alt, az) - 1.0)).round(1)
    return [[float(p) if np.isfinite(p) and (not model.coverage or covered(model.coverage, a, z)) else None
             for p, a, z in zip(prow, arow, zrow)] for prow, arow, zrow in zip(pct, alt, az)]
