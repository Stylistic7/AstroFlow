"""Star flats as first-class calibration masters.

A star flat is the detector's multiplicative response — vignetting, dust,
and the narrowband etalon a flat panel cannot reproduce — solved jointly from
the Gaia photometry of every plate-solved light subframe shot through one
(camera, focal length, filter) group (`stacks/starflat.py` for the math, the
`star_photometry` sidecar sections for the data). It is the DEFAULT flat for
every light stack of its group (`decisions/resolve.py`) unless the user
pins an explicit flat or "none".

Identity follows the Stack conventions exactly (attributes/chronology,
`Stack.from_frames`): `stable_hash` is the content hash of the oldest member
light, `content_hash` is the hash of the sorted member content hashes — so a
new sub in the group re-keys the star flat and, through `stack_files`'
`*cal_masters`, every stack that uses it. Membership is every solved light of
the group; the members' photometry sections are read at build time, and the
section schema is part of the identity so an engine change re-keys too.

Graph (compute_graph/compute_graph.py): `identify_star_flats` mints the
candidates above the spatial split (next to the camera fingerprints), one
`build_star_flat` task per candidate writes the master (embarrassingly
parallel across groups), `mint_star_flats` keeps the ones that pass the
quality gate, and `merge_star_flats` adds them to the calibration stacks
BEFORE any light↔flat association. Groups that fail the gate simply have no
star flat, and the run falls back to today's chronological/explicit default.
"""
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xxhash
import ray
import structlog
from beartype import beartype
from frozendict import frozendict

from attributes import ChronologyAttributes
from attributes.imaging_attributes import _camera_label
from cache.context_ref import ContextRef
from clustering.continuous import cluster_continuous
from decorators import node
from dependency_graph import Graph
from file_io.generic import read_header
from file_io.sidecar import get_section, set_section
from file_io.types import WithSidecar, XisfFilePath
from hashing import hash_list, hash_list_int
from metadata.classes import FrameHash, ImageMetadata, MetadataDict
from metadata.image_types import is_light_frame, is_master
from metadata.star_photometry import SCHEMA as PHOTOMETRY_SCHEMA
from stacks.composite_stack import CalibrationStack, CalibrationStacksDict, Stack, _camera_id_hash_from
from stacks.run import RunsDict
from stacks.star_flat_gain import (
    GAIN_BAND_PX, GAIN_LIMITS, GAIN_MAX_SUBS, GAIN_METHOD, GAIN_SCALE, apply_gain, fit_gain,
)
from stacks.starflat import (
    gather_star_photometry, response_image, solve_star_flat, split_half, write_master_flat,
)
from stacks.types import StackHash
from units import Celsius, Seconds

log = structlog.get_logger()

MIN_SUBS = 25                 # fewer solved lights than this: no candidate at all; fewer KEPT by
                              # the solve: built but not offered (the user's sparse-group rule)
MIN_SPLIT_HALF = 0.8          # odd/even-sub map agreement below this: built but not offered
CELL_UNCERTAINTY_MAX = 0.0015 # log units; the finest cell whose per-cell noise stays under 0.15%
GATE_UNCERTAINTY_MAX = 0.005  # log units; noisier than 0.5% per cell (even at 256 px) prints its
                              # cells into every stack — long-focal-length groups with few stars
                              # per sub sat at 0.6–1.1% on 2026-09-07; they fall back to the old
                              # defaults until more subs arrive
CELLS = (256, 512, 1024)      # fine-to-coarse candidates; the FINEST whose cells are all
                              # sufficiently sampled wins (`solve_adaptive`). 512/1024 exist so a
                              # star-poor long-focal group gets an honest coarse flat instead of a
                              # fine one fitted on a dozen stars per cell.
                              # 128 was tried in production (2026-09-12..14) and reverted: Stylistic7 judged
                              # its masters splotchier at medium scales than 256's; the per-stack gain
                              # (scale_star_flat) carries the amplitude, so 256 with fitted k is the choice.
                              # 128 px re-admitted 2026-09-12: the Cygnus strength x smoothing sweep
                              # (memory project_star_flat_undercorrects_by_scale) put the lowest etalon
                              # residual at 128 px / 0.7 cell on S and equal on H, with thousands of
                              # stars per cell; the uncertainty gate still drops sparse groups to 256.
                              # 64 px printed cell noise into masters (2026-09-07) and stays out.
# A cell's value is the weighted mean of its stars, so the cell — not the sub,
# and not the group — is where "enough data" has to be decided (Stylistic7,
# 2026-09-22). Stars alone are not enough: they must come from different
# POINTINGS, or a cell holds the same dozen stars measured over and over and
# their systematics become the flat.
MIN_CELL_STARS = 30           # stars in a cell before its gain means anything
MIN_CELL_POINTINGS = 4        # distinct sky positions contributing to that cell
CELL_COVERAGE_MIN = 0.98      # fraction of cells that must meet both, or try a coarser cell
MAG_WINDOW = (8.0, 13.5)      # forced photometry has no faint-star selection bias; SNR-weighted
SECTION = "star_flat"         # the master's sidecar section: quality metrics for the gate


@beartype
@dataclass(frozen=True)
class StarFlatStack(CalibrationStack):
    """A flat master whose "frames" are the group's light subframes.

    `contents` is empty (nothing to integrate); `member_files` are the
    lights whose sidecar photometry the solve reads."""
    focal_length: float | None = None
    member_files: tuple[str, ...] = ()
    photometry_schema: str = ""

    @property
    def kind(self) -> str:
        return "star"

    @property
    def n_subs(self) -> int:
        return len(self.member_files)

    def __hash__(self):
        return hash_list_int([self.content_hash])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CalibrationStack):
            return NotImplemented
        return (self.content_hash, self.stable_hash) == (other.content_hash, other.stable_hash)

    def __str__(self) -> str:
        return (f"StarFlatStack(Content hash {self.content_hash}, Camera {self.camera}, "
                f"Focal {self.focal_length}, Filter {self.filter}, Subs {self.n_subs})")


class StarFlats(dict[StackHash, StarFlatStack]):
    """The star flats that passed the gate, keyed by stable_hash."""
    def __hash__(self):
        return hash_list_int([self[k].content_hash for k in sorted(self.keys())])

    def __str__(self):
        return f"StarFlats({len(self)})"


class StarFlatCandidates(StarFlats):
    """Every group with enough solved lights, before the quality gate."""


def _focal_clusters(values) -> list[list[int]]:
    """Indices grouped by FOCALLEN within the matcher's 2% tolerance
    (`same_optical_train`), via the shared `cluster_continuous`.

    This was a hand-rolled copy of that clusterer which differed in one
    detail: it compared each value to the cluster's FIRST element rather
    than to the PREVIOUS one, so it was not single-link and its output
    depended on where a cluster happened to start. Real FOCALLEN values are
    a continuum, not discrete levels — one 350 mm rig in this library reports
    346, 347, 348, 349, 350, 351 and 352 — and single-link is what keeps such
    a run together.

    2% is a RELATIVE tolerance and `cluster_continuous` takes an absolute
    epsilon, so cluster in LOG space: a gap of log(1.02) between consecutive
    sorted values is exactly a 2% step.
    """
    return [
        [i for i, _ in cluster]
        for cluster in cluster_continuous(
            list(enumerate(values)),
            epsilon=math.log(1.02),
            key=lambda item: math.log(item[1]),
        )
    ]


def _group_content_hash(frames) -> str:
    """A star flat's content hash: its membership, plus the photometry schema.

    The membership part is the house convention — a Mosaic hashes its
    targets' content hashes, a Target its stacks', a Stack its runs', a Run
    its frames' — and a frame's content hash is its PIXELS, so the whole
    hierarchy is deliberately blind to metadata and re-solving a frame moves
    nothing. That is what keeps the library from re-deriving itself.

    The schema is the one extra term, the same shape as `DownscaledStack`'s
    `downscale_stack._fn_hash`: unlike a Stack, whose product depends only on
    its frames' pixels, a star flat's product depends on their PHOTOMETRY,
    which is a function of the pixels, the WCS and the measuring code. The
    schema names that code. Without it the 2026-09-21 re-solve changed every
    member's measurements and moved no hash, so `build_star_flat` cache-hit
    on flats built from the old solutions and every master kept them.

    NOT hashed: the individual plate solutions. That would re-key the group
    — and re-stack its whole train — every time a single sub was re-solved.
    See `MIN_SUBS`' note on the cost of a revision.
    """
    return hash_list(sorted(str(f) for f in frames) + [PHOTOMETRY_SCHEMA])


def star_flat_candidates(metadata: MetadataDict, fingerprints: dict) -> StarFlatCandidates:
    """One candidate per (camera identity, focal length, filter) with at
    least MIN_SUBS plate-solved lights. Camera identity is the defect-
    fingerprint hash (sensor geometry alone when a frame has no consensus),
    never INSTRUME — the same rule as every other stack."""
    groups: dict[tuple, list] = {}
    for h, m in metadata.items():
        header = m.header
        if not is_light_frame(header) or is_master(header) or m.plate_solution is None:
            continue
        filt = str(header.get("FILTER", "") or "").strip()
        focal = header.get("FOCALLEN")
        if not filt or focal in (None, ""):
            continue
        try:
            focal = float(focal)
        except (TypeError, ValueError):
            continue
        if focal <= 0:
            continue        # bad metadata; also keeps _focal_clusters' log() defined
        cam = _camera_id_hash_from(m, fingerprints.get(FrameHash(h), ())) or f"{m.width_pixels}x{m.height_pixels}"
        groups.setdefault((cam, filt), []).append((focal, FrameHash(h)))

    candidates = StarFlatCandidates()
    for (cam, filt), items in groups.items():
        for idx in _focal_clusters([f for f, _ in items]):
            if len(idx) < MIN_SUBS:
                continue
            frames = [items[i][1] for i in idx]
            sub = MetadataDict({f: metadata[f] for f in frames})
            chron = ChronologyAttributes.from_frames(frames, sub)
            first = metadata[frames[0]]
            stack = StarFlatStack(
                content_hash=_group_content_hash(frames),
                image_type="flat",
                camera=_camera_label(m.header.get("INSTRUME") for m in sub.values()),
                filter=filt,
                mean_temperature=Celsius(float(np.mean([float(m.header.get("CCD-TEMP", 0) or 0) for m in sub.values()]))),
                exposure=Seconds(1.0),
                integration=Seconds(0.0),
                width_pixels=first.width_pixels,
                height_pixels=first.height_pixels,
                stable_hash=chron.stable_hash,
                oldest_date=chron.oldest_date,
                newest_date=chron.newest_date,
                state=frozendict(),
                contents=RunsDict(),
                camera_id_hash=None if cam.count("x") == 1 and cam.replace("x", "").isdigit() else cam,
                focal_length=float(np.median([items[i][0] for i in idx])),
                member_files=tuple(sorted(str(metadata[f].filename) for f in frames)),
                photometry_schema=PHOTOMETRY_SCHEMA,
            )
            candidates[StackHash(stack.stable_hash)] = stack
    return candidates


@node
def identify_star_flats(
    metadata_ref: ContextRef[MetadataDict],
    fingerprint_map_ref: ContextRef,
) -> StarFlatCandidates:
    candidates = star_flat_candidates(metadata_ref.get(), fingerprint_map_ref.get())
    log.info("star flat candidates", groups=len(candidates),
             detail={k: (v.camera, v.focal_length, v.filter, v.n_subs) for k, v in candidates.items()})
    return candidates


def cell_coverage(sol, min_stars: int, min_pointings: int, pointings=None) -> dict:
    """How well `sol`'s cells are sampled: the fraction that carry at least
    `min_stars` stars, and — when `pointings` gives each sub's sky position —
    at least `min_pointings` DISTINCT pointings.

    The pointing count is what makes a cell's stars independent. Six nights
    on one galaxy put the same dozen stars in the same cell every time; the
    per-cell mean then converges on whatever those stars' systematics are,
    and split-half agreement is perfect because odd and even subs hold the
    SAME stars (2026-09-22: the 1234 mm L flat, split-half 0.996, carried a
    +15.7% blob no other filter on that sensor sees)."""
    n = sol.counts
    enough = n >= min_stars
    out = {"cells": int(n.size), "min_stars": min_stars,
           "covered_fraction": float(enough.mean()),
           "starved_cells": int((~enough).sum()),
           "median_stars_per_cell": float(np.median(n))}
    if pointings is not None:
        out["min_pointings"] = min_pointings
        out["pointing_covered_fraction"] = float((pointings >= min_pointings).mean())
        out["covered_fraction"] = float((enough & (pointings >= min_pointings)).mean())
    return out


def solve_adaptive(stars: np.ndarray, shape: tuple[int, int], cells=CELLS, mag_window=MAG_WINDOW,
                   min_stars=MIN_CELL_STARS, min_pointings=MIN_CELL_POINTINGS,
                   pointing_of_sub=None) -> tuple:
    """Solve from FINE to COARSE and keep the finest cell size at which every
    cell is sufficiently sampled — Stylistic7's rule (2026-09-22): "the star flat
    can only be built when all of its cells have sufficient samples".

    A sparse long-focal group then gets an honest coarse flat instead of a
    fine one fitted on a dozen stars per cell; a dense mosaic group still
    gets 256 px. When even the coarsest cell size leaves cells starved the
    caller gets `sol=None` and builds no flat at all.

    Returns (solution|None, cell|None, r_raw, r_blob, coverage).
    """
    worst = None
    for cell in sorted(cells, reverse=False):        # fine -> coarse
        sol = solve_star_flat(stars, shape, cell=cell, mag_window=mag_window)
        pts = _pointings_per_cell(stars, sol, pointing_of_sub) if pointing_of_sub is not None else None
        cov = cell_coverage(sol, min_stars, min_pointings, pts)
        cov["cell"] = cell
        if cov["covered_fraction"] >= CELL_COVERAGE_MIN:
            r_raw, r_blob = split_half(stars, sol)
            return sol, cell, r_raw, r_blob, cov
        worst = cov
    return None, None, float("nan"), float("nan"), (worst or {})


def _pointings_per_cell(stars: np.ndarray, sol, pointing_of_sub) -> np.ndarray:
    """Distinct pointings contributing to each cell, in `sol.counts`' shape."""
    H, W = sol.shape
    nx = -(-W // sol.cell)
    x, y, sid = stars[:, 0], stars[:, 1], stars[:, 10].astype(int)
    cells = (np.clip(y.astype(int), 0, H - 1) // sol.cell) * nx + (np.clip(x.astype(int), 0, W - 1) // sol.cell)
    pid = np.array([pointing_of_sub.get(int(i), -1) for i in sid])
    seen: dict[int, set] = {}
    for c, q in zip(cells, pid):
        seen.setdefault(int(c), set()).add(int(q))
    out = np.zeros(sol.counts.size, int)
    for c, qs in seen.items():
        if 0 <= c < out.size:
            out[c] = len(qs)
    return out.reshape(sol.counts.shape)


def passes_gate(quality: dict, *, min_subs: int, min_split_half: float, max_uncertainty: float) -> bool:
    """The rule a built master must meet to be offered as the default flat."""
    return (int(quality.get("n_subs", 0)) >= min_subs
            and float(quality.get("split_half_r", float("nan"))) >= min_split_half
            and float(quality.get("per_cell_uncertainty", float("inf"))) <= max_uncertainty)


def _passes(quality: dict) -> bool:
    return passes_gate(quality, min_subs=MIN_SUBS, min_split_half=MIN_SPLIT_HALF,
                       max_uncertainty=GATE_UNCERTAINTY_MAX)


# When a rebuilt flat counts as a DIFFERENT flat — publishing a new identity
# and re-stacking its whole train, deliberately, once — rather than the same
# one re-solved. Both must hold (2026-10-08, Stylistic7: "if the star flats really
# are getting better, I really do want to restack the world with them"):
#   significant: the RMS of the relative change over the field is at least
#     STAR_FLAT_SIGNIFICANCE times the two solves' combined per-cell
#     uncertainty — a change the measurement can actually distinguish from
#     re-solved noise;
#   material: that RMS change exceeds STAR_FLAT_CHANGE — big enough to matter
#     against the 0.3-1% patterns these flats correct.
# An RMS over the field hides a change confined to one region (a new dust
# shadow, a corner the old solve never covered), so a LOCAL change also
# publishes: the largest relative change anywhere at least
# STAR_FLAT_LOCAL_CHANGE and STAR_FLAT_LOCAL_SIGNIFICANCE times the noise.
# Measured on the two cases that motivated this: the RedCat OIII flat re-solved
# with 5% more subs changed 0.04% RMS at 0.45x its uncertainty (the same flat;
# carry forward) while the young FLT156 OIII flat changed 1.1% RMS at 2.7x with
# its uncertainty down 3x (a better flat; re-stack). Until then the gate was
# the MAX over the field against 0.2%: with 35 cells at 0.06% uncertainty the
# maximum crossed it on noise alone whenever a night's subs were added, and the
# RedCat train (100 stacks, 276 h) re-stacked on consecutive mornings.
STAR_FLAT_CHANGE = 0.002
STAR_FLAT_SIGNIFICANCE = 2.0
STAR_FLAT_LOCAL_CHANGE = 0.01
STAR_FLAT_LOCAL_SIGNIFICANCE = 5.0


def _unit(response: np.ndarray) -> np.ndarray:
    """`response` about its own median — the form the flats are compared in
    (`write_master_flat` stores them at a median of 0.5)."""
    return np.asarray(response, np.float64) / float(np.nanmedian(response))


def response_identity(response: np.ndarray) -> str:
    """A star flat's identity: a hash of the pixels it will be written as.
    Computed here, at write time, from the array already in hand — never by
    re-reading the file (`hash_of_image_data`'s tripwire exists for that)."""
    from hashing import hash_list
    r = (_unit(response) * 0.5).astype(np.float32)
    return hash_list([xxhash.xxh64(np.ascontiguousarray(r)).hexdigest()])


def _standing_flat(dest: Path, star_flat: StarFlatStack) -> tuple[np.ndarray, str, dict] | None:
    """The flat this group most recently published — its pixels and its
    declared identity — or None when it has never built one.

    Found by scanning this op's own output directory for a file stamped with
    the group's STBLHASH: the same lookup `gate()` does over built masters,
    and the only stable handle a star flat has (its stable_hash is the oldest
    member's, which never moves as the group grows). Star flats are not
    pipeline display taps, so nothing supersedes them and earlier revisions
    stay on disk for exactly this comparison.

    Only the header is read for the match; the pixels of the ONE match are
    read after."""
    from file_io.generic import read_declared_content_hash, read_image
    best: tuple[float, Path] | None = None
    for path in Path(dest).parent.glob("*.xisf"):
        if path == Path(dest):
            continue
        try:
            if str(read_header(path).get("STBLHASH", "")) != str(star_flat.stable_hash):
                continue
            mtime = path.stat().st_mtime
        except (OSError, ValueError, KeyError):
            continue
        if best is None or mtime > best[0]:
            best = (mtime, path)
    if best is None:
        return None
    identity = read_declared_content_hash(best[1])
    if identity is None:
        return None
    return (np.asarray(read_image(best[1]), np.float32).squeeze(), identity,
            get_section(best[1], SECTION) or {})


def _pointing_of_sub(members: list[Path]) -> dict[int, int]:
    """Each member's sky position, as an integer id per ~0.1 deg cell — the
    unit `solve_adaptive` counts to tell independent samples from repeats.
    Read from the sidecar's plate solution; a member without one gets its own
    id, which can only make a cell look LESS independent than it is."""
    from file_io.sidecar import get_section
    out: dict[int, int] = {}
    ids: dict[tuple, int] = {}
    for i, m in enumerate(members):
        ps = get_section(m, "plate_solution") or {}
        c = ((ps.get("data") or {}).get("center")) if isinstance(ps, dict) else None
        key = (round(float(c[0]), 1), round(float(c[1]), 1)) if c else ("unsolved", i)
        out[i] = ids.setdefault(key, len(ids))
    return out


def solve_and_write(star_flat: StarFlatStack, dest, *, cells=CELLS, mag_window=MAG_WINDOW) -> dict:
    """Solve the group, write `dest` as a WBPP-ready Master Flat XISF
    carrying the star flat's STBLHASH, and record the quality metrics in
    its sidecar (`SECTION`); `good` there is informational (UI, logs) — the
    gate itself is applied in `mint_star_flats`, whose thresholds sit in its
    cache key. Returns the metrics."""
    members = [Path(p) for p in star_flat.member_files]
    stars = gather_star_photometry(members)
    n_with = int(len(np.unique(stars[:, 10]))) if len(stars) else 0
    shape = (star_flat.height_pixels, star_flat.width_pixels)
    log.info("building star flat", stack=str(star_flat), subs_with_photometry=n_with, stars=len(stars))
    if len(stars) == 0 or n_with == 0:
        # Nothing to solve. `solve_star_flat` normalises its weights by their
        # own max, so an empty star table reaches numpy as `max()` of a
        # zero-size array and the op dies — which then fails `mint_star_flats`
        # and `build_regroup_graph` above it, and the whole pass ends with a
        # FAILED node instead of a flat of ones (boot 97, 2026-09-22: five
        # groups whose members yielded no photometry stalled the rebuild for
        # hours). The guard used to live in the `n_with < MIN_SUBS` branch
        # that the per-cell sufficiency rule replaced.
        response = np.ones(shape, np.float32)
        quality = {"n_subs": n_with, "stars": 0, "good": False,
                   "reason": "no member sub yielded any photometry"}
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        write_master_flat(response, dest, {
            "IMAGETYP": "Master Flat", "FILTER": star_flat.filter or "", "INSTRUME": star_flat.camera,
            "FOCALLEN": star_flat.focal_length, "XBINNING": 1, "YBINNING": 1, "EXPOSURE": 1.0, "EXPTIME": 1.0,
            "STBLHASH": star_flat.stable_hash, "SFLATSUB": 0, "SFLATCEL": 0,
            "CONTHASH": response_identity(response),
        })
        set_section(Path(dest), SECTION, quality)
        log.info("star flat built", stack=str(star_flat), **quality)
        return quality
    sol, cell, r_raw, r_blob, coverage = solve_adaptive(
        stars, shape, cells, mag_window, pointing_of_sub=_pointing_of_sub(members))
    if sol is None:
        # No cell size samples the sensor well enough. A flat of ones is the
        # honest answer — WBPP divides by it and the stack falls back to its
        # panel flat — and the reason rides in the sidecar and the log.
        response = np.ones(shape, np.float32)
        quality = {"n_subs": n_with, "stars": int(len(stars)), **coverage,
                   "reason": ("cells under-sampled at every cell size: "
                              f"{coverage.get('covered_fraction', 0):.0%} covered at "
                              f"{coverage.get('cell', '?')} px, need {CELL_COVERAGE_MIN:.0%} "
                              f"with >= {MIN_CELL_STARS} stars from >= {MIN_CELL_POINTINGS} pointings")}
    else:
        response = response_image(sol, smooth_sigma_cells=0.7)
        quality = {"n_subs": len(sol.kept_subs), "cell": cell, "split_half_r": r_raw, "split_half_blob": r_blob,
                   "per_cell_uncertainty": float(np.nanmedian(sol.uncertainty)), "colour_term": sol.colour_term,
                   "residual_scatter": sol.residual_scatter, **coverage}
    quality["good"] = sol is not None and _passes(quality)
    # DECLARE AN IDENTITY (see `file_io.generic`'s CALLHASH/CONTHASH note).
    # A star flat is shared by every stack of its optical train, so without
    # this, adding subs to ONE stack moved the group's membership, moved this
    # op's call hash, and re-stacked the whole train through WBPP for a flat
    # that had not meaningfully changed. When the new response matches the
    # standing flat within `STAR_FLAT_CHANGE`, we write the STANDING flat's
    # pixels — so the declared identity is literally the same content, and
    # every consumer's key holds still.
    standing = _standing_flat(dest, star_flat)
    identity = None
    if standing is not None and standing[0].shape == response.shape:
        old_response, old_identity, old_quality = standing
        delta = _unit(response) / _unit(old_response) - 1.0
        change = float(np.sqrt(np.nanmean(np.square(delta))))
        local = float(np.nanmax(np.abs(delta)))
        unc_new = float(quality.get("per_cell_uncertainty") or 0.0)
        unc_old = float(old_quality.get("per_cell_uncertainty") or 0.0)
        noise = float(np.hypot(unc_new, unc_old))
        pattern = float(np.sqrt(np.nanmean(np.square(_unit(old_response) - 1.0))))
        quality.update(
            change_vs_standing=change,
            change_significance=change / noise if noise > 0 else float("inf"),
            change_vs_pattern=change / pattern if pattern > 0 else float("inf"),
            change_local=local,
            change_local_significance=local / noise if noise > 0 else float("inf"),
            uncertainty_ratio=unc_new / unc_old if unc_old > 0 else None,
        )
        different = quality["change_significance"] >= STAR_FLAT_SIGNIFICANCE and change > STAR_FLAT_CHANGE
        local_change = (quality["change_local_significance"] >= STAR_FLAT_LOCAL_SIGNIFICANCE
                        and local >= STAR_FLAT_LOCAL_CHANGE)
        if not (different or local_change):
            response, identity = old_response, old_identity
            quality["carried_forward"] = True
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    write_master_flat(response, dest, {
        "IMAGETYP": "Master Flat", "FILTER": star_flat.filter or "", "INSTRUME": star_flat.camera,
        "FOCALLEN": star_flat.focal_length, "XBINNING": 1, "YBINNING": 1, "EXPOSURE": 1.0, "EXPTIME": 1.0,
        "STBLHASH": star_flat.stable_hash, "SFLATSUB": quality.get("n_subs", 0), "SFLATCEL": quality.get("cell", 0),
        "CONTHASH": identity or response_identity(response),
    })
    set_section(Path(dest), SECTION, quality)
    log.info("star flat built", stack=str(star_flat), **quality)
    return quality


# `display=` is what makes this a user-facing PRODUCT rather than an
# intermediate: it sets `makes_tiff`, so `_publish_with_presentations`
# writes the tile-pyramid sibling every other master gets (stack_files
# declares display="Master"). Without it a star flat was stamped and
# nothing else, so its detail page had no pyramid to serve and the
# 61 MP primary was refused by the thumbnail budget -> silent 404
# (2026-09-09). Decorator lists are stripped from the structural hash,
# so adding this does NOT re-key the 91 existing masters.
@node(num_cpus=4, display="Star Flat", resources={"diskcache": 0.001}, declares_identity=True)
def build_star_flat(star_flat: StarFlatStack, cells: tuple, mag_window: tuple) -> WithSidecar[XisfFilePath]:
    """The group's master flat (see `solve_and_write`). The solve settings
    are arguments so they sit in the cache key: a new cell size or window
    rebuilds every master by itself.

    Head-pinned (`diskcache`): the build reads ~1,000 member sidecars, and a
    freshly joined worker's share reports existing files as missing for a
    while — boot 58 (2026-09-07) built 37 groups on the MBP with
    subs_with_photometry=0. A missing member is an error here, never "no
    photometry": the task fails and is retried instead of minting a blank."""
    missing = [p for p in star_flat.member_files if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {star_flat.n_subs} member subs unreadable here, e.g. {missing[0]}")
    dest = build_star_flat.filename_fn(star_flat, cells, mag_window)
    solve_and_write(star_flat, dest, cells=cells, mag_window=mag_window)
    return WithSidecar(XisfFilePath(dest))


GAIN_SECTION = "star_flat_gain"   # the scaled master's sidecar section: the fit, for the record


@node(num_cpus=2, ephemeral=True, resources={"diskcache": 0.001})
def scale_star_flat(stack: Stack, star_flat: ImageMetadata, dark: ImageMetadata | None,
                    band: tuple, scale: int, max_subs: int, method: int,
                    kind: str = "star") -> WithSidecar[XisfFilePath]:
    """The star flat re-scaled to THIS stack's sky: 1 + k * (F - 1), with k fitted
    on the stack's own subs in sensor coordinates (`stacks.star_flat_gain`).
    Handed to WBPP in place of the group's star flat, so the per-frame
    division — and every meridian flip — is handled exactly as before.

    Ephemeral: a 61 MP float32 per stack that is a few seconds of arithmetic
    on the group's flat; reclaimed once `stack_files` has consumed it. The
    fit settings are arguments so they sit in the cache key. Head-pinned for
    the same reason as `build_star_flat`: it reads the stack's raw subs.
    The fitted k and per-sub slopes go to the master's sidecar (`GAIN_SECTION`)
    and to a Concern when the gain is unusual.

    `star_flat` / `dark` arrive as `ImageMetadata` — a producer's result reaches
    its consumer through the metadata wrap, exactly as `stack_files` receives
    its `*cal_masters` — and `@node` enforces the hint (boot 81 failed all 286
    calls on a `WithSidecar` hint)."""
    from concerns import warn
    from file_io.generic import read_image
    from file_io.xisf_file_io import read_xisf_header
    paths = sorted(str(m.filename) for m in stack.all_metadata().values())
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(paths)} subs unreadable here, e.g. {missing[0]}")
    flat = read_image(Path(star_flat.filename))
    dark_img = read_image(Path(dark.filename)) if dark is not None else None
    gain_range, unusual = GAIN_LIMITS[kind]
    fit = fit_gain(paths, flat, dark_img, band=band, scale=scale, max_subs=max_subs,
                   gain_range=gain_range)
    k = fit["k"]
    log.info("star flat gain", stack=str(stack), k=round(k, 3), n_used=fit["n_used"],
             median_corr=round(float(np.median(fit["corrs"])), 2) if fit["corrs"] else None,
             reason=fit.get("reason"))
    if fit.get("withheld"):
        warn("Star flat withheld — it does not match this stack's sky",
             f"{fit['reason']}. Applying it would print its own structure into the master "
             f"(2026-09-22: at k = 1 this punched holes into the galaxy masters). The stack "
             f"falls back to its panel or sky flat; the star flat for this group may be built "
             f"from too few pointings to describe the sensor.",
             stack=stack.stable_hash, **{k_: v for k_, v in fit.items() if k_ not in ("slopes", "corrs")})
    elif fit.get("reason"):
        warn("Star flat gain could not be fitted; applied as built (k = 1)",
             f"{fit['reason']}. The stack's sky did not yield a believable amplitude for the "
             f"star flat's etalon pattern, so the group's flat is used unchanged.",
             stack=stack.stable_hash, **{k_: v for k_, v in fit.items() if k_ not in ("slopes", "corrs")})
    elif not (unusual[0] <= k <= unusual[1]):
        warn(f"Star flat gain k = {k:.2f} is unusual",
             "The stack's sky sees the sensor's etalon pattern at an amplitude far from what "
             "its neighbours do — often thin cloud or moonlight changing the sky's spectrum. "
             "The flat is applied at this gain; check the subs' transparency.",
             stack=stack.stable_hash, k=round(k, 3), n_used=fit["n_used"])
    dest = scale_star_flat.filename_fn(stack, star_flat, dark, band, scale, max_subs, method)
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    # The group's identity keywords ride along — WBPP matches the master to
    # the stack by STBLHASH (`str`: the XISF reader literal_evals an all-digit
    # hash into an int) — and IMAGETYP is written afresh: `normalize_header`
    # rewrites it, and WBPP recognises a pre-built master only as 'Master Flat'.
    hdr = read_xisf_header(Path(star_flat.filename))
    keywords = {key: hdr[key] for key in ("FILTER", "INSTRUME", "FOCALLEN", "XBINNING", "YBINNING",
                                          "EXPOSURE", "EXPTIME", "SFLATSUB", "SFLATCEL") if key in hdr}
    keywords.update({"IMAGETYP": "Master Flat", "STBLHASH": str(hdr.get("STBLHASH", "")),
                     "SFLATK": round(k, 4)})
    write_master_flat(apply_gain(flat, k), dest, keywords)
    set_section(Path(dest), GAIN_SECTION, fit)
    return WithSidecar(XisfFilePath(dest))


def gain_args(stack: Stack, flat, dark, kind: str = "star") -> tuple:
    """`scale_star_flat`'s arguments for `stack` — the fit settings ride in
    the cache key the same way `build_args` puts the solve settings there.
    `kind` selects the gain limits (`GAIN_LIMITS`): "star" or "sky"."""
    return (stack, flat, dark, GAIN_BAND_PX, GAIN_SCALE, GAIN_MAX_SUBS, GAIN_METHOD, kind)


def build_args(star_flat: StarFlatStack) -> tuple:
    """`build_star_flat`'s full argument tuple for this star flat — every
    caller (graph, calibrate, previews) must key the same way."""
    return (star_flat, CELLS, MAG_WINDOW)


def gate(candidates: StarFlatCandidates, master_paths, *, min_subs: int, min_split_half: float,
         max_uncertainty: float) -> StarFlats:
    """The candidates whose built master's recorded metrics pass the rule."""
    out = StarFlats()
    for path in master_paths:
        sid = StackHash(str(read_header(Path(path))["STBLHASH"]))
        q = get_section(Path(path), SECTION) or {}
        if sid in candidates and passes_gate(q, min_subs=min_subs, min_split_half=min_split_half,
                                             max_uncertainty=max_uncertainty):
            out[sid] = candidates[sid]
        else:
            log.info("star flat not offered", stack=sid, quality=q)
    return out


@node
def mint_star_flats(candidates: StarFlatCandidates, min_subs: int, min_split_half: float,
                    max_uncertainty: float, *masters: ImageMetadata) -> StarFlats:
    """The thresholds are arguments (not module constants read inside) so
    that tightening the rule re-keys this node by itself; the masters stay
    cached. A producer's output reaches its consumers as the written file's
    ImageMetadata (boot 59 failed here with a WithSidecar hint)."""
    return gate(candidates, [Path(m.filename) for m in masters], min_subs=min_subs,
                min_split_half=min_split_half, max_uncertainty=max_uncertainty)


@ray.remote(num_cpus=0.1)
def scatter_star_flats(candidates: StarFlatCandidates) -> Graph:
    """One build per candidate, then the gate. The subgraph's result (the
    StarFlats) flows on to the spatial-group scatter as an ancestor."""
    builds = [(build_star_flat, *build_args(candidates[k])) for k in sorted(candidates)]
    graph = Graph({b: {} for b in builds})
    collector = (mint_star_flats, candidates, MIN_SUBS, MIN_SPLIT_HALF, GATE_UNCERTAINTY_MAX)
    graph[collector] = builds                            # ordered: value-consuming collector
    return graph


@node
def merge_star_flats(star_flats: StarFlats, cal_stacks: CalibrationStacksDict) -> CalibrationStacksDict:
    """Star flats join the calibration stacks before cal-to-cal association
    and every light↔flat resolution."""
    return CalibrationStacksDict({**cal_stacks, **star_flats})


def star_flat_master_path(star_flat: StarFlatStack) -> Path | None:
    """The built master's path when it exists in the cache, else None
    (previews must never trigger a build)."""
    hit, value = build_star_flat.cached_value(*build_args(star_flat))
    if not hit or value is None:
        return None
    # A producer's cached value is the written file's ImageMetadata, not the
    # WithSidecar it returned; `.path` raised AttributeError on every cache
    # HIT while a MISS silently returned None, so the preview either 500'd or
    # quietly served an uncalibrated frame (2026-09-08).
    return Path(value.filename)
