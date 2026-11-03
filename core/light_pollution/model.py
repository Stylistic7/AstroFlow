"""One site's sky glow as a small model, and what it says about any frame.

**City glow.** Light pollution is light domes over the cities around a site:
brightness rises toward the horizon, differently in each direction. The
model is log10 brightness relative to the zenith = (X−1)·F1(az) + (X−1)²·F2(az),
X the airmass, F1 and F2 Fourier series in azimuth to HARMONICS — N_CITY
numbers, `theta`. It is fitted in PERCENT of each frame's sky level per degree,
on the tilt CHANGES within runs (a run's nebula cancels; a night's brightness
cancels), from moonless dark frames. Measured 2026-11-02 on 231 San Francisco
runs: 74 % of the within-run changes explained; MDW-scored stacks 0.81 of the
uncorrected gradient (GraXpert 0.94, and 3.2 on the nebula-rich b98d).

**Moonlight.** Scattered moonlight has a known shape (Krisciunas & Schaefer
1991: separation from the Moon, phase, the Moon's airmass, the line of
sight's) and unknown amplitude. Per filter two amplitudes in counts per
second, `city` and `moon`, are fitted in ABSOLUTE units on all dark runs,
moonlit included, under non-negativity. Same day: moonlit runs 46 % → 80 %
explained; moonlit-heavy stacks 0.81 → 0.70.

**Two regimes from one theta.** With the Moon down a frame's surface is its
sky level × (P/⟨P⟩ − 1), P = 10^(basis·theta): the percent form absorbs the
night's brightness. With the Moon up, the level is no longer the glow, so the
surface is exposure × [city·(P/⟨P⟩ − 1) + moon·(ks − ⟨ks⟩)], the full
moonlight field per cell. A filter with too few moonlit runs keeps moon = 0
and is simply not Moon-corrected; the report says so.

**Identity.** `content_hash` is of theta and the amplitudes alone (the
MoteModel rule): counts and fit quality are the record. The fit keeps the
previous model, counts refreshed, unless the new one's predictions differ
by more than HYSTERESIS — otherwise every night's runs would re-key every
integration. This fold is path-dependent by design.

Imports nothing from compute_graph or stacks: `compute_graph.objects` holds a
`SiteModels` and imports this module at import time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from frozendict import frozendict

from hashing import hash_list
from light_pollution.robust import irls, irls_bounded
from light_pollution.sites import Site, same_site

if TYPE_CHECKING:
    from light_pollution.measure import RunTilts

HARMONICS = 3
N_CITY = 2 * (2 * HARMONICS + 1)
K_EXT = 0.15                      # magnitudes per airmass, the extinction K&S scatter against
MOON_UP_DEG = -5.0                # above this the Moon lights the sky; below, the moonless regime
MIN_RUNS = 20                     # moonless dark runs a shape needs
MIN_EXPLAINED = 0.4               # of the within-run tilt changes
MIN_MOON_FRAMES = 150             # moonlit frames a filter's moon amplitude needs
FRAME_GATE_PCT_PER_DEG = 0.2      # below this a frame's surface is noise: not subtracted
HYSTERESIS_PCT_PER_DEG = 0.15     # a refit replaces the stored model only beyond this (refits of one
HYSTERESIS_AMPLITUDE = 0.2        # sky with fresh noise differ by ~0.13); relative, per amplitude
ROUND = 4
REFERENCE_ALT = np.arange(30.0, 86.0, 5.0)   # where frames are shot; below, the (X-1)^2 terms amplify noise
REFERENCE_AZ = np.arange(0.0, 360.0, 15.0)
# A model is trusted only where the site's frames are: a Fourier dome does what it likes elsewhere
# (Oakland, 2026-11-02: 45 runs over three octants predicted -100 % glow to the south). Coverage is
# counted in cells of COVERAGE_ALT_STEP x COVERAGE_AZ_STEP degrees with at least COVERAGE_FRAMES dark frames.
COVERAGE_ALT_STEP = 15.0
COVERAGE_AZ_STEP = 45.0
COVERAGE_FRAMES = 5
MAX_DARKER_THAN_ZENITH = 0.1      # a shape darker than the zenith by more than this inside its coverage is unphysical


def coverage_cell(alt_deg: float, az_deg: float) -> tuple[int, int]:
    return int(min(alt_deg, 89.9) // COVERAGE_ALT_STEP), int((az_deg % 360.0) // COVERAGE_AZ_STEP)


def coverage_of(tilts) -> tuple[tuple[int, int], ...]:
    counts: dict[tuple[int, int], int] = {}
    for t in tilts:
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
    content_hash: str = ""

    @classmethod
    def of(cls, site: Site, theta, amplitudes, **record) -> "SiteModel":
        theta = tuple(round(float(t), ROUND) for t in theta)
        amplitudes = tuple(sorted((str(f), round(float(c), ROUND), round(float(m), ROUND)) for f, c, m in amplitudes))
        return cls(site, theta, amplitudes, content_hash=hash_list([theta, amplitudes]), **record)

    def amplitude(self, filt: str) -> tuple[float, float] | None:
        return next(((c, m) for f, c, m in self.amplitudes if f == filt), None)

    def summary(self) -> dict:
        return {"hash": self.content_hash, "runs": self.n_runs, "frames": self.n_frames,
                "explained": round(self.explained, 3), "explained_moonlit": round(self.explained_moonlit, 3),
                "newest_run": self.newest_run, "coverage": [list(c) for c in self.coverage],
                "amplitudes": {f: {"city": c, "moon": m} for f, c, m in self.amplitudes}}


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
def _within_run(tilts, rows_of, y_of, x_of):
    """Stack within-run differences: for each run, the chosen frames' (y, x) minus the run's mean."""
    Y, X = [], []
    n_runs = n_frames = 0
    for t in tilts:
        idx = rows_of(t)
        if len(idx) < 2:
            continue
        y = np.array([y_of(t, j) for j in idx], float)
        x = np.array([x_of(t, j) for j in idx], float)
        Y.append((y - y.mean(0)).reshape(-1))
        X.append((x - x.mean(0)).reshape(-1, x.shape[-1]))
        n_runs += 1
        n_frames += len(idx)
    if not Y:
        return np.zeros(0), np.zeros((0, 1)), 0, 0
    return np.concatenate(Y), np.concatenate(X), n_runs, n_frames


def _explained(coef, X, Y, inl) -> float:
    """The share of the (robust) variance the model explains: MAD-based, so a
    cloud-crossed run cannot make a good model look useless."""
    if inl.sum() < 4:
        return 0.0
    y, r = Y[inl], Y[inl] - X[inl] @ coef
    spread = 1.4826 * np.median(np.abs(y - np.median(y)))
    if spread == 0:
        return 0.0
    return float(1 - (1.4826 * np.median(np.abs(r - np.median(r))) / spread) ** 2)


def _reference_tilts(theta) -> np.ndarray:
    """The shape's gradient along altitude, %/°, over the reference grid — what two models are compared by."""
    alt, az = np.meshgrid(REFERENCE_ALT, REFERENCE_AZ, indexing="ij")
    up = 100.0 * math.log(10.0) * (basis(alt + 0.5, az) @ np.asarray(theta))
    down = 100.0 * math.log(10.0) * (basis(alt - 0.5, az) @ np.asarray(theta))
    return up - down


def _same_within_hysteresis(new: SiteModel, previous: SiteModel) -> bool:
    d = _reference_tilts(new.theta) - _reference_tilts(previous.theta)
    if float(np.sqrt(np.mean(d ** 2))) > HYSTERESIS_PCT_PER_DEG:
        return False
    old = dict((f, (c, m)) for f, c, m in previous.amplitudes)
    for f, c, m in new.amplitudes:
        oc, om = old.get(f, (0.0, 0.0))
        for a, b in ((c, oc), (m, om)):
            if abs(a - b) > HYSTERESIS_AMPLITUDE * max(abs(a), abs(b), 1e-9) and abs(a - b) > 1e-3:
                return False
    return set(old) == {f for f, _, _ in new.amplitudes}


MAX_SAMPLES = 2000


def _samples(tilts) -> list:
    """(alt, az, moonless) of the dark frames, at most MAX_SAMPLES — the panel's dots."""
    pts = [(round(t.alt[j], 1), round(t.az[j], 1), bool(t.moonless[j]))
           for t in tilts for j in range(t.n_frames) if t.dark[j]]
    return pts[::max(1, -(-len(pts) // MAX_SAMPLES))]


def fit_site_model(site: Site, tilts, previous: SiteModel | None) -> tuple[SiteModel | None, dict]:
    """The site's model from its runs' tilts (`light_pollution.measure.RunTilts`),
    and the site's report; (None, report) when the data do not support a model."""
    tilts = [t for t in tilts if t is not None]
    dark = lambda t: [j for j in range(t.n_frames) if t.dark[j]]                          # noqa: E731
    moonless = lambda t: [j for j in range(t.n_frames) if t.dark[j] and t.moonless[j]]  # noqa: E731
    report = {"key": site.key, "lat": site.lat, "lon": site.lon,
              "runs": len(tilts), "frames": sum(t.n_frames for t in tilts),
              "usable_frames": sum(len(dark(t)) for t in tilts),
              "moonlit_frames": sum(len(dark(t)) - len(moonless(t)) for t in tilts),
              "newest_run": max((t.date for t in tilts), default=""),
              "samples": _samples(tilts), "model": None, "reason": ""}
    # 1. the shape, in percent of the sky level, from moonless frames
    Y, X, n_runs, n_frames = _within_run(tilts, moonless, lambda t, j: t.tilt_pct[j], lambda t, j: t.city_pct[j])
    if n_runs < MIN_RUNS:
        report["reason"] = f"needs at least {MIN_RUNS} moonless dark runs; has {n_runs}"
        return None, report
    theta, inl, _ = irls(X, Y)
    explained = _explained(theta, X, Y, inl)
    if explained < MIN_EXPLAINED:
        report["reason"] = f"the shape explains only {100 * explained:.0f}% of the tilt changes within runs"
        return None, report
    coverage = coverage_of(tilts)
    darkest = min((float(city_shape(theta, (a + 0.5) * COVERAGE_ALT_STEP, (z + 0.5) * COVERAGE_AZ_STEP))
                   for a, z in coverage), default=1.0)
    if darkest < 1.0 - MAX_DARKER_THAN_ZENITH:
        report["reason"] = (f"the shape is {100 * (1 - darkest):.0f}% darker than the zenith inside the site's own "
                            f"coverage: unphysical, the runs do not pin it down")
        return None, report
    # 2. the amplitudes, in counts per second, per filter, from all dark frames
    amplitudes = []
    for filt in sorted({t.filter for t in tilts}):
        of_filter = [t for t in tilts if t.filter == filt]
        moonlit = sum(len(dark(t)) - len(moonless(t)) for t in of_filter)
        with_moon = moonlit >= MIN_MOON_FRAMES
        Ya, Xa, runs_a, _ = _within_run(
            of_filter, dark, lambda t, j: t.tilt_abs[j],
            lambda t, j: np.column_stack([np.asarray(t.city_pct[j]) @ theta / 100.0,
                                          np.asarray(t.moon[j])])[:, :2 if with_moon else 1])
        if runs_a < 3:
            continue
        coef, _, _ = irls_bounded(Xa, Ya)
        amplitudes.append((filt, float(coef[0]), float(coef[1]) if with_moon else 0.0))
    # 3. how well the whole model does on moonlit frames
    amp = {f: (c, m) for f, c, m in amplitudes}
    Ym, Xm, _, _ = _within_run(
        [t for t in tilts if t.filter in amp], lambda t: [j for j in dark(t) if not t.moonless[j]],
        lambda t, j: t.tilt_abs[j],
        lambda t, j: amp[t.filter][0] * (np.asarray(t.city_pct[j]) @ theta / 100.0)[:, None]
        + amp[t.filter][1] * np.asarray(t.moon[j])[:, None])
    explained_moonlit = _explained(np.ones(1), Xm, Ym, np.ones(len(Ym), bool)) if len(Ym) else 0.0
    record = dict(n_runs=n_runs, n_frames=n_frames, explained=explained,
                  explained_moonlit=explained_moonlit, newest_run=report["newest_run"], coverage=coverage)
    model = SiteModel.of(site, theta, amplitudes, **record)
    if previous is not None and _same_within_hysteresis(model, previous):
        model = replace(previous, **record)
    report["model"] = model.summary()
    return model, report


# ------------------------------------------------------------------ a frame's surface
def frame_surface(model: SiteModel, header: dict, wcs_header: dict, shape: tuple[int, int],
                  level: float, filt: str, step: int = 64):
    """The glow in this frame's own pixel grid at its mid-exposure moment, as
    (surface in counts with mean 0, or None; its tilt in % of `level` per
    degree; the regime: 'moonless', 'moonlit', 'gated', 'no-amplitude' or
    'uncovered' — outside the directions the site's frames sampled).
    `level` is the frame's sky level in counts (lower-half median of its
    block medians); `header` carries DATE-OBS, EXPTIME, SITELAT/SITELONG."""
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
        amp = model.amplitude(filt)
        if amp is None:
            return None, 0.0, "no-amplitude"
        phase = 180.0 - float(moon.separation(get_sun(t)).deg)
        ks = moonlight(cells.separation(moon).deg, aa.alt.deg, moon_alt, phase)
        surface = float(header["EXPTIME"]) * (amp[0] * city + amp[1] * (ks - ks.mean()))
        regime = "moonlit"
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


def brightness_grid(model: SiteModel, alts=np.arange(15.0, 91.0, 5.0), azs=np.arange(0.0, 360.0, 10.0)) -> list:
    """Percent brighter than the zenith, rows by altitude, columns by azimuth — the panel's map.
    None outside the model's coverage, where the shape is not trusted (`frame_surface` corrects nothing there)."""
    alt, az = np.meshgrid(alts, azs, indexing="ij")
    pct = (100.0 * (city_shape(model.theta, alt, az) - 1.0)).round(1)
    return [[float(p) if np.isfinite(p) and (not model.coverage or covered(model.coverage, a, z)) else None
             for p, a, z in zip(prow, arow, zrow)] for prow, arow, zrow in zip(pct, alt, az)]
