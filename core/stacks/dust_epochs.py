"""Dust-mote EPOCHS of one optical system, folded from the dust maps of its
own science frames (`stacks/dust_runs.py` measures one map per night and
pointing; this module folds them). The result assigns every dust run a
`MoteModel`: the motes that epoch's lights see, which `with_scaled_flats`
multiplies into a SYNTHETIC flat (a star or night-sky flat is epoch-free: it
was built mote-free or shot on another night). A real panel/twilight flat is
never touched — its motes are supposed to be there.

Design (Stylistic7, 2026-10-13):
  * dust is fixed in pixel coordinates and the sky is not, so a mote is a
    dark, round, compact dip that repeats at the same pixel across DIFFERENT
    pointings; a dark lane in one field is not;
  * an epoch opens at a BOUNDARY: a run in which a known mote is decidedly
    absent (the dust moved — the 2026-10-04 camera rotation), or the first
    run of a mote that then persists (dust landed). It opens with the
    SURVIVORS, the previous motes that are still there;
  * while an epoch is PROVISIONAL its model is re-fitted as pointings accrue
    and every stack with a run in it re-keys: the correction is improving.
    Once the model's split-half error — fitted on the odd pointings against
    the even ones — is below TOLERANCE the epoch is FINAL and frozen, so the
    stacks of a closed epoch never move again. One pointing can never
    converge (one half is empty), which is the conservative outcome;
  * a model's identity is its quantized motes alone — not the epoch, not
    its state — so finalising, and re-fits below the quantum, re-key nothing.

Everything here is a pure function of the run maps in date order: the fold
is recomputed from scratch on every graph pass (seconds on cached maps) and
gives a closed epoch byte-identical entries whatever arrives after it.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field, astuple

import numpy as np
import structlog
from scipy import ndimage

from hashing import hash_list
from stacks.flat_motes import find_motes, radial_profile

log = structlog.get_logger()

TOLERANCE = 0.005        # the one knob: a correction error below this is invisible (fraction of sky)
QUANTUM = TOLERANCE / 2  # a profile value is rounded to this ...
POS_QUANTUM = 2.0        # ... a centre to this many px, an extent to one block
POINTING_SEP_DEG = 0.25  # closer than this is the same pointing (dithers are ~0.02 deg)
FLATTEN_PX = 256.0       # the detector's scale: what is smoother than this is sky, vignetting, etalon
DECIDE_SIGMA = 4.0       # present/absent in a run only this far from the half-depth line
STEP = 0.5               # profile radius step, blocks (what `radial_profile` fits at)
MIN_SHADOW = 0.5         # a rendered shadow never falls below this: dividing by it gains at most
                         # 2x; a mote darker than that is an occultation, not a correctable shadow
MIN_SURVIVOR_DEPTH = 0.005
TRAILING = 10            # a landing is also sought on the newest pointings alone: a whole epoch's
                         # median hides a change that is still a minority of its pointings


@dataclass(frozen=True)
class Mote:
    """One mote: centre and extent in full-res px, and its shadow as the dip
    (1 - S) per STEP blocks of radius from the centre, 0 beyond the extent."""
    x: float
    y: float
    extent: float
    profile: tuple[float, ...]

    @property
    def depth(self) -> float:
        return float(np.mean(self.profile[:2])) if self.profile else 0.0

    def radius(self, block: int) -> float:
        """Half-depth radius, px — the disc a run is scored over."""
        half = self.depth / 2
        i = next((i for i, p in enumerate(self.profile) if p < half), len(self.profile))
        return max(i * STEP, 1.0) * block


@dataclass(frozen=True)
class MoteModel:
    """The motes of one epoch. `content_hash` is of the motes ALONE: the epoch
    and its state are the record, not the identity, so a stacking key moves
    only when a mote does."""
    motes: tuple[Mote, ...]
    epoch: str = ""
    final: bool = False
    error: float | None = None
    content_hash: str = ""

    @classmethod
    def of(cls, motes, *, epoch: str = "", final: bool = False, error: float | None = None) -> "MoteModel":
        motes = tuple(sorted(motes, key=lambda m: (m.y, m.x)))
        return cls(motes, epoch, final, error, hash_list([astuple(m) for m in motes]) if motes else "none")

    def summary(self) -> dict:
        return {"epoch": self.epoch, "final": self.final, "error": self.error, "hash": self.content_hash,
                "motes": [{"x": m.x, "y": m.y, "extent": m.extent, "depth": round(m.depth, 4)} for m in self.motes]}


@dataclass
class RunMap:
    """One run's measurement: the block image (1.0 = flat) of its subs."""
    run: str
    date: _dt.datetime
    centre: tuple[float, float]
    n_subs: int
    map: np.ndarray


@dataclass
class Epoch:
    start: int                       # index of the run it opens at
    runs: list[int] = field(default_factory=list)
    model: MoteModel = field(default_factory=lambda: MoteModel.of(()))
    final: bool = False
    pointings: int = 0


# ------------------------------------------------------------------ geometry
def assign_pointings(runs: list[RunMap]) -> list[int]:
    """Pointing ids in run order: a run joins the first pointing within
    POINTING_SEP_DEG of its centre, else opens one. Deterministic in the
    runs' order, which is their date order."""
    centres: list[tuple[float, float]] = []
    ids = []
    for r in runs:
        ra, dec = r.centre
        for i, (ra0, dec0) in enumerate(centres):
            dra = (ra - ra0 + 180.0) % 360.0 - 180.0
            if np.hypot(dra * np.cos(np.radians(dec0)), dec - dec0) < POINTING_SEP_DEG:
                ids.append(i)
                break
        else:
            centres.append((ra, dec))
            ids.append(len(centres) - 1)
    return ids


def _dip(map_b: np.ndarray, block: int) -> tuple[np.ndarray, float]:
    """The map's dip below its own smooth (sky, vignetting, etalon), and the
    per-block noise of that dip."""
    m = np.nan_to_num(map_b, nan=1.0).astype(np.float32)
    dip = 1.0 - m / np.maximum(ndimage.gaussian_filter(m, FLATTEN_PX / block), 1e-6)
    sigma = 1.4826 * float(np.median(np.abs(dip - np.median(dip)))) or 1e-6
    return dip, sigma


def presence(dip: np.ndarray, sigma: float, mote: Mote, block: int) -> int:
    """+1 the mote is in this map, -1 it is decidedly not, 0 cannot say."""
    r = mote.radius(block) / block
    yy, xx = np.ogrid[: dip.shape[0], : dip.shape[1]]
    disc = (xx - mote.x / block) ** 2 + (yy - mote.y / block) ** 2 <= r * r
    n = int(disc.sum())
    if n == 0:
        return 0
    resp = float(dip[disc].astype(np.float32).mean())
    margin = DECIDE_SIGMA * sigma / np.sqrt(n)
    half = mote.depth / 2
    if resp - margin > half:
        return 1
    if resp + margin < half:
        return -1
    return 0


# ------------------------------------------------------------------ fitting
def _quantized(cx: float, cy: float, ext: float, prof: np.ndarray, block: int) -> Mote:
    n = int(ext / STEP) + 1
    p = np.clip(np.asarray(prof[:n], np.float64), 0, None)
    return Mote(x=round(cx * block / POS_QUANTUM) * POS_QUANTUM,
                y=round(cy * block / POS_QUANTUM) * POS_QUANTUM,
                extent=float(max(1, round(ext)) * block),
                profile=tuple(float(round(v / QUANTUM) * QUANTUM) for v in p))


def _split_error(odd: np.ndarray, even: np.ndarray, mote: Mote, block: int) -> float:
    """How much the odd pointings' shadow disagrees with the even ones': the
    RMS difference of the two profiles over the mote's area (a ring's weight
    is its radius — the innermost ring is one noisy block and must not
    decide alone)."""
    r0 = max(mote.radius(block) / block, 2.0)
    po, _, _ = radial_profile(odd, mote.x / block, mote.y / block, r0, recentre=False)
    pe, _, _ = radial_profile(even, mote.x / block, mote.y / block, r0, recentre=False)
    n = len(mote.profile)
    po, pe = np.clip(po[:n], 0, None), np.clip(pe[:n], 0, None)
    k = min(len(po), len(pe))
    if not k:
        return 1.0
    w = np.arange(k) + 0.5
    return float(np.sqrt(np.sum(w * (po[:k] - pe[:k]) ** 2) / w.sum()))


def fit_model(pooled: np.ndarray, odd: np.ndarray | None, even: np.ndarray | None,
              survivors: tuple[Mote, ...], block: int, epoch: str) -> MoteModel:
    """The motes of `pooled` (the epoch's pointings, medianed): the
    detector's candidates and the survivors, each fitted as a radial shadow
    at its centroid. A candidate is admitted only when it is in both halves
    and its split-half error is under half its depth — correcting is then
    better than not correcting. The model's `error` is the worst split of
    any candidate, admitted or not: a mote still too noisy holds the epoch
    open."""
    cands = [(m["x"] / block, m["y"] / block, max(m["radius_px"] / block, 2.0)) for m in find_motes(pooled, block)]
    for s in survivors:
        if not any(abs(s.x / block - x) < 3 and abs(s.y / block - y) < 3 for x, y, _ in cands):
            cands.append((s.x / block, s.y / block, max(s.radius(block) / block, 2.0)))
    halves = odd is not None and even is not None
    if not halves:
        # one pointing: nothing new can be told from the sky, and the
        # survivors are known better from their old epoch than from one
        # field — the model is exactly them, so nothing re-keys yet
        return MoteModel.of(survivors, epoch=epoch, error=None)
    dips = [_dip(h, block) for h in (odd, even)]
    admitted, errors = [], []
    for x, y, r0 in cands:
        prof, ext, (cx, cy) = radial_profile(pooled, x, y, r0)
        mote = _quantized(cx, cy, ext, prof, block)
        survivor = any(abs(s.x - mote.x) < 3 * block and abs(s.y - mote.y) < 3 * block for s in survivors)
        if mote.depth < (MIN_SURVIVOR_DEPTH if survivor else 0.01):
            continue
        if any(presence(d, s, mote, block) < 0 for d, s in dips):
            continue
        if any(abs(a.x - mote.x) < 3 * block and abs(a.y - mote.y) < 3 * block for a in admitted):
            continue              # a detector hit and a survivor centroid onto the same mote
        err = _split_error(odd, even, mote, block)
        errors.append(err)        # a candidate still too noisy to admit also holds the epoch open
        if err >= mote.depth / 2:
            continue
        admitted.append(mote)
    return MoteModel.of(admitted, epoch=epoch, error=max(errors) if errors else 0.0)


# ------------------------------------------------------------------ the fold
class _Pool:
    """Per-pointing means of the run maps (nan-aware), medianed across
    pointings on demand; odd/even halves by pointing order."""

    def __init__(self, shape: tuple[int, int]):
        self.sum: dict[int, np.ndarray] = {}
        self.cnt: dict[int, np.ndarray] = {}
        self.order: list[int] = []
        self.shape = shape

    def add(self, p: int, m: np.ndarray) -> None:
        if p not in self.sum:
            self.sum[p] = np.zeros(self.shape, np.float64)
            self.cnt[p] = np.zeros(self.shape, np.int32)
            self.order.append(p)
        ok = np.isfinite(m)
        self.sum[p][ok] += m[ok]
        self.cnt[p][ok] += 1

    def _mean(self, p: int) -> np.ndarray:
        return np.where(self.cnt[p] > 0, self.sum[p] / np.maximum(self.cnt[p], 1), np.nan).astype(np.float32)

    def pooled(self, points: list[int]) -> np.ndarray | None:
        if not points:
            return None
        # a block no run of a pointing measured is flat (1.0) for the median:
        # np.median partitions, np.nanmedian sorts — 3x slower on 50 maps
        stack = np.stack([np.nan_to_num(self._mean(p), nan=1.0) for p in points])
        return np.median(stack, axis=0).astype(np.float32)

    def halves(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self.pooled(self.order[0::2]), self.pooled(self.order[1::2])


def _landing_index(dips: list, members: list[int], mote: Mote, block: int) -> int:
    """The run a newly found mote landed on: the one after the LAST run of
    the epoch where it is decidedly absent. Never decidedly absent means it
    was there all along and the epoch itself refits (one restack: the data
    now proves a mote its early pointings could not show)."""
    last = next((i for i in reversed(members) if presence(dips[i][0], dips[i][1], mote, block) < 0), None)
    if last is None:
        return members[0]
    k = members.index(last) + 1
    return members[min(k, len(members) - 1)]


def _near(a: Mote, b: Mote, block: int) -> bool:
    return abs(a.x - b.x) < 3 * block and abs(a.y - b.y) < 3 * block


def fold(system: str, runs: list[RunMap], block: int) -> list[Epoch]:
    """The epochs of one rig and FILTER from its runs in date order, each
    with its model. Filters are independent (Stylistic7, 2026-10-13): a camera
    opened for one filter changes only that filter's dust, and a filter shot
    rarely simply catches up once it has a few pointings since a change —
    no cross-filter bookkeeping."""
    if not runs:
        return []
    pointings = assign_pointings(runs)
    dips = [(d.astype(np.float16), s) for d, s in (_dip(r.map, block) for r in runs)]
    shape = runs[0].map.shape
    epochs: list[Epoch] = []
    cur: Epoch | None = None
    pool: _Pool | None = None
    checked = 0                      # pointings in the pool at the last landing check
    scores: dict[tuple, int] = {}    # (run, mote) -> presence, never changing: memoised

    def score(i: int, m: Mote) -> int:
        key = (i, m.x, m.y, m.radius(block))
        if key not in scores:
            scores[key] = presence(dips[i][0], dips[i][1], m, block)
        return scores[key]

    def open_epoch(at: int, survivors: tuple[Mote, ...]) -> None:
        nonlocal cur, pool, checked
        cur = Epoch(start=at, model=MoteModel.of(survivors, epoch=f"{system}:{runs[at].run[:8]}"))
        pool = _Pool(shape)
        checked = 0

    def refit() -> None:
        assert cur is not None and pool is not None
        odd, even = pool.halves()
        before = cur.model
        model = fit_model(pool.pooled(pool.order), odd, even, before.motes, block, before.epoch)
        if odd is not None and even is not None and model.error is not None and model.error < TOLERANCE:
            cur.final = True
            model = MoteModel.of(model.motes, epoch=model.epoch, final=True, error=model.error)
            log.info("dust epoch final", system=system, epoch=model.epoch, runs=len(cur.runs),
                     pointings=len(pool.order), error=model.error, motes=model.summary()["motes"])
        if model.content_hash != before.content_hash:
            log.info("dust model refit", system=system, epoch=before.epoch, runs=len(cur.runs),
                     pointings=len(pool.order), was=before.summary()["motes"], now=model.summary()["motes"],
                     error=model.error)
        cur.model = model

    i = 0
    while i < len(runs):
        run = runs[i]
        # only a FINAL model's motes can vanish: a provisional one is still
        # being refitted, and its marginal motes come and go with the noise
        # (the FLT156 R row split into a dozen one-run epochs without this)
        if cur is not None and cur.final and cur.model.motes:
            gone = [m for m in cur.model.motes if score(i, m) < 0]
            if gone:
                survivors = tuple(m for m in cur.model.motes if score(i, m) > 0)
                log.info("dust epoch boundary: motes vanished", system=system, run=run.run[:8], date=str(run.date),
                         vanished=[(m.x, m.y) for m in gone], survivors=[(m.x, m.y) for m in survivors],
                         closed=cur.model.summary())
                epochs.append(cur)
                open_epoch(i, survivors)
        if cur is None:
            open_epoch(i, ())
        assert cur is not None and pool is not None
        cur.runs.append(i)
        new_pointing = pointings[i] not in pool.order
        pool.add(pointings[i], run.map)
        cur.pointings = len(pool.order)
        last = i == len(runs) - 1
        if not cur.final:
            if new_pointing or last:
                refit()
        elif (len(pool.order) - checked >= 2 or last) and len(pool.order) >= 3:
            # a landing check costs two refits (~3 s): only with two new
            # pointings to show, and always on the newest run. A landing is a
            # mote the frozen model lacks that a refit ADMITS (both halves,
            # split error under half its depth) — on the whole epoch, or on
            # its newest pointings alone, since a whole epoch's median hides
            # a change still a minority of its pointings
            checked = len(pool.order)
            known = cur.model.motes
            found = list(fit_model(pool.pooled(pool.order), *pool.halves(), known, block, cur.model.epoch).motes)
            tail = pool.order[-TRAILING:]
            if 2 <= len(tail) < len(pool.order):
                found += fit_model(pool.pooled(tail), pool.pooled(tail[0::2]), pool.pooled(tail[1::2]),
                                   known, block, cur.model.epoch).motes
            new = [m for m in found if not any(_near(m, k, block) for k in known)]
            if new:
                mote = new[0]
                j = _landing_index(dips, cur.runs, mote, block)
                log.info("dust epoch boundary: mote landed", system=system, at=runs[j].run[:8], date=str(runs[j].date),
                         mote=(mote.x, mote.y, round(mote.depth, 4)), epoch=cur.model.epoch)
                if j == cur.start:
                    cur.final = False          # it was always there: the epoch refits itself
                    refit()
                else:
                    cur.runs = [k for k in cur.runs if k < j]
                    cur.pointings = len({pointings[k] for k in cur.runs})
                    epochs.append(cur)
                    open_epoch(j, known)
                    i = j                      # replay the runs from the landing into the new epoch
                    continue
        i += 1
    if cur is not None:
        epochs.append(cur)
    return epochs


# ------------------------------------------------------------------ rendering
def render_motes(model: MoteModel, shape: tuple[int, int], block: int) -> np.ndarray:
    """The multiplicative mote map M (<= 1) of `model` at full resolution:
    each mote's radial shadow about its centre, 1 beyond its extent."""
    out = np.ones(shape, np.float32)
    for m in model.motes:
        if not m.profile:
            continue
        r = int(m.extent) + 1
        y0, y1 = max(0, int(m.y) - r), min(shape[0], int(m.y) + r + 1)
        x0, x1 = max(0, int(m.x) - r), min(shape[1], int(m.x) + r + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.sqrt((xx - m.x) ** 2 + (yy - m.y) ** 2) / (STEP * block)
        p = np.interp(d, np.arange(len(m.profile)), m.profile, right=0.0)
        out[y0:y1, x0:x1] *= np.maximum(1.0 - p, MIN_SHADOW).astype(np.float32)
    return out
