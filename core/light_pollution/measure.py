"""One run's subframes -> what the site model is fitted from.

Within a run every frame looks at the same sky, so the difference between a
frame and the run's median holds no nebula — only how the glow changed as the
field moved across the dome and the Moon moved across the sky (plus noise,
clouds, a satellite: the fits are robust). Each frame's tilt is measured
against that median, in percent of the frame's sky level per degree and in
counts per second per degree; beside it, the tilts the city shape's terms and
the moonlight field would have at that frame's moment, so the fit is linear.

Calibration: the OFFSET pedestal, as the dust maps' is (`stacks/dust_runs.py`),
and the run's flat — the night-sky flat is the etalon correction, and it comes
BEFORE any estimate of the glow (Stylistic7, 2026-11-02). The flat is `block_flat`:
the flat stack's own frames block-medianed to BIN px and normalised, a cached
node per flat stack, so the measurement needs nothing a pipeline makes — the
model exists before any pipeline runs and for all of them. Frames are
block-medianed to BIN px (71″ on the RedCat), which drops stars and hot pixels,
then resampled onto the first frame's grid through their plate solutions. The
sky level is the lower-half median of the block medians: an estimate under the
nebula, used again when a frame's surface is scaled.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog

from decorators import node
from hashing import hash_list
from light_pollution.model import MOON_UP_DEG, basis, moonlight
from light_pollution.robust import irls, lmeds, mad_scale
from light_pollution.sites import same_site, site_of
from light_pollution.time import mid_exposure
from stacks.dust_runs import PEDESTAL_PER_OFFSET
from stacks.run import Run
from stacks.sky_flat_plans import DARK_SUN_ALT_DEG

log = structlog.get_logger()

BIN = 32          # px per block
COARSE = 7        # every COARSE-th block samples the smooth shapes
MIN_FRAMES = 3


@dataclass(frozen=True)
class RunTilts:
    """One run's measurement: per frame, tuples only (it rides in cache keys)."""
    run: str
    position: tuple[float, float]
    filter: str
    date: str
    n_frames: int
    frames: tuple[str, ...]
    tilt_pct: tuple[tuple[float, float], ...]                         # %/deg of the sky level
    tilt_abs: tuple[tuple[float, float], ...]                         # counts/s/deg
    city_pct: tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]   # (2, N_CITY): the shape terms' tilts, %/deg
    moon: tuple[tuple[float, float], ...]                             # the moonlight field's tilt, per deg
    level: tuple[float, ...]
    alt: tuple[float, ...]
    az: tuple[float, ...]
    sun_alt: tuple[float, ...]
    moon_alt: tuple[float, ...]
    phase: tuple[float, ...]
    dark: tuple[bool, ...]
    moonless: tuple[bool, ...]
    content_hash: str = ""

    @classmethod
    def of(cls, **fields) -> "RunTilts":
        identity = hash_list([fields["run"], fields["position"], fields["tilt_pct"], fields["tilt_abs"],
                              fields["city_pct"], fields["moon"]])
        return cls(**fields, content_hash=identity)

    @classmethod
    def empty(cls, run: Run) -> "RunTilts":
        e = ()
        return cls(run.content_hash, (0.0, 0.0), "", "", 0, e, e, e, e, e, e, e, e, e, e, e, e, e, hash_list([run.content_hash, "empty"]))


@dataclass(frozen=True)
class FlatBlock:
    """A flat stack's shape at BIN resolution, normalised to a median of 1.
    Its identity is the flat stack's: the image follows from its frames."""
    flat: str
    kind: str
    n_frames: int
    image: np.ndarray
    content_hash: str = ""

    @classmethod
    def of(cls, flat: str, kind: str, n_frames: int, image: np.ndarray) -> "FlatBlock":
        return cls(flat, kind, n_frames, image, hash_list([flat, "block", BIN]))


def block_median(a: np.ndarray, b: int = BIN) -> np.ndarray:
    h, w = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    v = a[:h, :w].reshape(h // b, b, w // b, b).transpose(0, 2, 1, 3).reshape(h // b, w // b, b * b)
    return np.nanmedian(v, axis=2).astype(np.float32)


MAX_FLAT_FRAMES = 24


def flat_block(files) -> np.ndarray:
    """The flat's shape from its frames: each pedestal-subtracted, block-medianed
    (stars and hot pixels drop out) and normalised to its own median, the median
    across frames, normalised to 1. No-data cells are 1."""
    from file_io.generic import read_header, read_image
    files = list(files)
    files = files[::max(1, -(-len(files) // MAX_FLAT_FRAMES))]
    blocks = []
    for f in files:
        pedestal = PEDESTAL_PER_OFFSET * float(read_header(Path(f)).get("OFFSET", 0) or 0)
        b = block_median(np.asarray(read_image(Path(f)), np.float32).squeeze() - pedestal)
        blocks.append(b / np.nanmedian(b))
    flat = np.nanmedian(np.stack(blocks), 0)
    flat = flat / np.nanmedian(flat)
    return np.where(np.isfinite(flat) & (flat > 0), flat, 1.0).astype(np.float32)


def flat_files(flat_stack) -> list[str]:
    """A night-sky flat's members, or a flat stack's frames."""
    members = getattr(flat_stack, "member_files", ())
    if members:
        return [str(f) for f in members]
    return [str(m.filename) for run in flat_stack.contents.values() for m in run.metadata.values()]


@node(num_cpus=1, resources={"diskcache": 0.001})
def block_flat(flat_stack) -> FlatBlock:
    """Head-pinned like every reader of raw frames."""
    files = flat_files(flat_stack)
    image = flat_block(files)
    log.info("flat blocked", flat=flat_stack.content_hash[:8], kind=getattr(flat_stack, "kind", "flat"),
             frames=len(files), shape=image.shape)
    return FlatBlock.of(flat_stack.content_hash, str(getattr(flat_stack, "kind", "flat")), len(files), image)


def sky_level(d: np.ndarray) -> float:
    v = d[np.isfinite(d)]
    return float(np.median(v[v <= np.median(v)])) if v.size else float("nan")


def _wcs(solution_header: dict):
    from astropy.io import fits
    from astropy.wcs import WCS
    return WCS(fits.Header({k: v for k, v in solution_header.items()
                            if isinstance(v, (int, float, str)) and k not in ("COMMENT", "HISTORY")}))


def _plane(values: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Least-squares plane tilts of smooth model fields (no outliers): (2, k)."""
    return np.linalg.lstsq(A, values, rcond=None)[0][1:]


def _structure(a: np.ndarray, cubic: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """`a` (raveled) minus its robust cubic surface: sky-fixed structure, not glow."""
    coef, _, _ = irls(cubic[ok], a[ok])
    return a - cubic @ coef


MIN_STRUCTURED_CELLS = 200


def transparency(d: np.ndarray, median: np.ndarray, cubic: np.ndarray, ok: np.ndarray) -> tuple[float, float]:
    """The frame's scale and offset against the run median, (s, c), with s from
    the STRUCTURE both share (nebula, star residue) and never from their smooth
    parts: a scale fitted on the whole frame absorbs a glow change collinear
    with the median's own gradient, which is exactly the change to measure.
    Without enough structure the scale is 1 (a run's frames share a sky)."""
    d, median = d.ravel(), median.ravel()
    hp_m, hp_d = _structure(median, cubic, ok), _structure(d, cubic, ok)
    strong = ok & (np.abs(hp_m) > 3 * mad_scale(hp_m[ok]))
    s = 1.0
    if strong.sum() >= MIN_STRUCTURED_CELLS:
        (s,), _, _ = lmeds(hp_m[strong][:, None], hp_d[strong])
        s = float(np.clip(s, 0.3, 3.0))
    c = float(np.median((d - s * median)[ok]))
    return s, c


def run_tilts(run: Run, flat: FlatBlock | None = None) -> RunTilts:
    """The measurement, or `RunTilts.empty` when fewer than MIN_FRAMES solved
    frames state the same position. With `flat`, every frame is divided by it
    before anything is measured."""
    from astropy import units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, get_sun
    from astropy.time import Time
    from astropy.utils import iers
    from astropy.wcs.utils import proj_plane_pixel_scales
    from scipy.ndimage import map_coordinates

    from file_io.generic import read_image
    iers.conf.auto_download = False    # a worker may be offline; arcsecond accuracy is plenty here
    iers.conf.auto_max_age = None

    frames = [(h, m) for h, m in run.metadata.items()
              if m.plate_solution is not None and site_of(m.header) is not None]
    if len(frames) < MIN_FRAMES:
        return RunTilts.empty(run)
    frames.sort(key=lambda hm: mid_exposure(hm[1].header))
    position = site_of(frames[0][1].header)
    frames = [(h, m) for h, m in frames if same_site(site_of(m.header), position)]
    if len(frames) < MIN_FRAMES:
        return RunTilts.empty(run)

    ref = _wcs(frames[0][1].plate_solution.header)
    scale_deg = float(proj_plane_pixel_scales(ref)[0]) * BIN
    D = []
    for k, (h, m) in enumerate(frames):
        pedestal = PEDESTAL_PER_OFFSET * float(m.header.get("OFFSET", 0) or 0)
        img = block_median(np.asarray(read_image(Path(m.filename)), np.float32).squeeze() - pedestal)
        if flat is not None:
            h_, w_ = min(img.shape[0], flat.image.shape[0]), min(img.shape[1], flat.image.shape[1])
            img = img[:h_, :w_] / flat.image[:h_, :w_]
        if k == 0:
            shape = img.shape
            cy, cx = np.mgrid[0:shape[0], 0:shape[1]]
            px, py = cx * BIN + (BIN - 1) / 2, cy * BIN + (BIN - 1) / 2
            ra, dec = ref.all_pix2world(px.ravel(), py.ravel(), 0)
            D.append(img)
        else:
            x, y = _wcs(m.plate_solution.header).all_world2pix(ra, dec, 0, quiet=True)
            D.append(map_coordinates(img, [(y - (BIN - 1) / 2) / BIN, (x - (BIN - 1) / 2) / BIN],
                                     order=1, cval=np.nan).reshape(shape).astype(np.float32))
    D = np.stack(D)
    median = np.nanmedian(D, 0)
    X, Y = (cx - cx.mean()) * scale_deg, (cy - cy.mean()) * scale_deg
    xn, yn = X / np.abs(X).max(), Y / np.abs(Y).max()
    cubic = np.stack([np.ones(xn.size)] + [(xn ** i * yn ** j).ravel() for i in range(4) for j in range(4 - i) if i + j], 1)
    coarse = SkyCoord(ra[::COARSE] * u.deg, dec[::COARSE] * u.deg)
    A_coarse = np.stack([np.ones(coarse.size), X.ravel()[::COARSE], Y.ravel()[::COARSE]], 1)
    lat, lon = position
    location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg)

    out = {k: [] for k in ("tilt_pct", "tilt_abs", "city_pct", "moon", "level", "alt", "az",
                           "sun_alt", "moon_alt", "phase", "dark", "moonless")}
    for d, (h, m) in zip(D, frames):
        ok = np.isfinite(d) & np.isfinite(median)
        s, c = transparency(d, median, cubic, ok.ravel())
        r = d - s * median - c
        g, _, _ = lmeds(np.stack([np.ones(ok.sum()), X[ok], Y[ok]], 1), r[ok])
        level = sky_level(d)
        exposure = float(m.header.get("EXPTIME") or 1.0)
        t = Time(mid_exposure(m.header), scale="utc")
        frame = AltAz(obstime=t, location=location)
        aa = coarse.transform_to(frame)
        city = 100.0 * math.log(10.0) * _plane(basis(aa.alt.deg, aa.az.deg), A_coarse)
        sun_alt = float(get_sun(t).transform_to(frame).alt.deg)
        moon = get_body("moon", t, location)
        moon_alt = float(moon.transform_to(frame).alt.deg)
        phase = 180.0 - float(moon.separation(get_sun(t)).deg)
        if moon_alt > MOON_UP_DEG:
            ks = moonlight(coarse.separation(moon).deg, aa.alt.deg, moon_alt, phase)
            gm = _plane(ks, A_coarse)
        else:
            gm = np.zeros(2)
        out["tilt_pct"].append(tuple(float(v) for v in 100.0 * g[1:] / level))
        out["tilt_abs"].append(tuple(float(v) for v in g[1:] / exposure))
        out["city_pct"].append(tuple(tuple(float(v) for v in row) for row in city))
        out["moon"].append(tuple(float(v) for v in gm))
        out["level"].append(level)
        out["alt"].append(float(np.median(aa.alt.deg)))
        out["az"].append(float(np.degrees(math.atan2(np.median(np.sin(aa.az.rad)), np.median(np.cos(aa.az.rad)))) % 360))
        out["sun_alt"].append(sun_alt)
        out["moon_alt"].append(moon_alt)
        out["phase"].append(phase)
        out["dark"].append(sun_alt < DARK_SUN_ALT_DEG)
        out["moonless"].append(moon_alt < MOON_UP_DEG)
    return RunTilts.of(run=run.content_hash, position=position, filter=str(frames[0][1].header.get("FILTER", "")),
                       date=mid_exposure(frames[0][1].header).isoformat(), n_frames=len(frames),
                       frames=tuple(h for h, _ in frames), **{k: tuple(v) for k, v in out.items()})


@node(num_cpus=1, resources={"diskcache": 0.001})
def measure_run_tilts(run: Run, *flat: FlatBlock) -> RunTilts:
    """Head-pinned like every reader of raw subs (`measure_dust_run`). `flat`
    is the run's flat as a `block_flat` result when the run resolves to one."""
    tilts = run_tilts(run, flat[0] if flat else None)
    log.info("run tilts measured", run=run.content_hash[:8], frames=tilts.n_frames,
             site=tilts.position, filter=tilts.filter, flat=flat[0].flat[:8] if flat else None)
    return tilts
