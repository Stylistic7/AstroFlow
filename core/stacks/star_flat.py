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
    GAIN_BAND_PX, GAIN_MAX_SUBS, GAIN_METHOD, GAIN_SCALE, GAIN_UNUSUAL, apply_gain, fit_gain,
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
CELLS = (256,)                # coarse-to-fine candidates; the finest under CELL_UNCERTAINTY_MAX wins.
                              # 128 was tried in production (2026-09-12..14) and reverted: Stylistic7 judged
                              # its masters splotchier at medium scales than 256's; the per-stack gain
                              # (scale_star_flat) carries the amplitude, so 256 with fitted k is the choice.
                              # 128 px re-admitted 2026-09-12: the Cygnus strength x smoothing sweep
                              # (memory project_star_flat_undercorrects_by_scale) put the lowest etalon
                              # residual at 128 px / 0.7 cell on S and equal on H, with thousands of
                              # stars per cell; the uncertainty gate still drops sparse groups to 256.
                              # 64 px printed cell noise into masters (2026-09-07) and stays out.
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
                content_hash=hash_list(sorted(str(f) for f in frames)),
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


def solve_adaptive(stars: np.ndarray, shape: tuple[int, int], cells=CELLS, mag_window=MAG_WINDOW) -> tuple:
    """Solve at `cells` from coarse to fine and keep the finest cell whose
    median per-cell uncertainty stays under CELL_UNCERTAINTY_MAX.
    Returns (solution, cell, r_raw, r_blob)."""
    best = None
    for cell in cells:
        sol = solve_star_flat(stars, shape, cell=cell, mag_window=mag_window)
        unc = float(np.nanmedian(sol.uncertainty)) if np.isfinite(sol.uncertainty).any() else float("inf")
        if best is None or unc <= CELL_UNCERTAINTY_MAX:
            best = (sol, cell)
        if unc > CELL_UNCERTAINTY_MAX:
            break
    sol, cell = best
    r_raw, r_blob = split_half(stars, sol)
    return sol, cell, r_raw, r_blob


def passes_gate(quality: dict, *, min_subs: int, min_split_half: float, max_uncertainty: float) -> bool:
    """The rule a built master must meet to be offered as the default flat."""
    return (int(quality.get("n_subs", 0)) >= min_subs
            and float(quality.get("split_half_r", float("nan"))) >= min_split_half
            and float(quality.get("per_cell_uncertainty", float("inf"))) <= max_uncertainty)


def _passes(quality: dict) -> bool:
    return passes_gate(quality, min_subs=MIN_SUBS, min_split_half=MIN_SPLIT_HALF,
                       max_uncertainty=GATE_UNCERTAINTY_MAX)


def solve_and_write(star_flat: StarFlatStack, dest, *, cells=CELLS, mag_window=MAG_WINDOW) -> dict:
    """Solve the group, write `dest` as a WBPP-ready Master Flat XISF
    carrying the star flat's STBLHASH, and record the quality metrics in
    its sidecar (`SECTION`); `good` there is informational (UI, logs) — the
    gate itself is applied in `mint_star_flats`, whose thresholds sit in its
    cache key. Returns the metrics."""
    stars = gather_star_photometry([Path(p) for p in star_flat.member_files])
    n_with = int(len(np.unique(stars[:, 10]))) if len(stars) else 0
    shape = (star_flat.height_pixels, star_flat.width_pixels)
    log.info("building star flat", stack=str(star_flat), subs_with_photometry=n_with, stars=len(stars))
    if n_with < MIN_SUBS:
        response = np.ones(shape, np.float32)
        quality = {"n_subs": n_with, "reason": "too few subs with photometry"}
    else:
        sol, cell, r_raw, r_blob = solve_adaptive(stars, shape, cells, mag_window)
        response = response_image(sol, smooth_sigma_cells=0.7)
        quality = {"n_subs": len(sol.kept_subs), "cell": cell, "split_half_r": r_raw, "split_half_blob": r_blob,
                   "per_cell_uncertainty": float(np.nanmedian(sol.uncertainty)), "colour_term": sol.colour_term,
                   "residual_scatter": sol.residual_scatter}
    quality["good"] = _passes(quality)
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    write_master_flat(response, dest, {
        "IMAGETYP": "Master Flat", "FILTER": star_flat.filter or "", "INSTRUME": star_flat.camera,
        "FOCALLEN": star_flat.focal_length, "XBINNING": 1, "YBINNING": 1, "EXPOSURE": 1.0, "EXPTIME": 1.0,
        "STBLHASH": star_flat.stable_hash, "SFLATSUB": quality.get("n_subs", 0), "SFLATCEL": quality.get("cell", 0),
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
@node(num_cpus=4, display="Star Flat", resources={"diskcache": 0.001})
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
                    band: tuple, scale: int, max_subs: int, method: int) -> WithSidecar[XisfFilePath]:
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
    fit = fit_gain(paths, flat, dark_img, band=band, scale=scale, max_subs=max_subs)
    k = fit["k"]
    log.info("star flat gain", stack=str(stack), k=round(k, 3), n_used=fit["n_used"],
             median_corr=round(float(np.median(fit["corrs"])), 2) if fit["corrs"] else None,
             reason=fit.get("reason"))
    if fit.get("reason"):
        warn("Star flat gain could not be fitted; applied as built (k = 1)",
             f"{fit['reason']}. The stack's sky did not yield a believable amplitude for the "
             f"star flat's etalon pattern, so the group's flat is used unchanged.",
             stack=stack.stable_hash, **{k_: v for k_, v in fit.items() if k_ not in ("slopes", "corrs")})
    elif not (GAIN_UNUSUAL[0] <= k <= GAIN_UNUSUAL[1]):
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


def gain_args(stack: Stack, star_flat, dark) -> tuple:
    """`scale_star_flat`'s arguments for `stack` — the fit settings ride in
    the cache key the same way `build_args` puts the solve settings there."""
    return (stack, star_flat, dark, GAIN_BAND_PX, GAIN_SCALE, GAIN_MAX_SUBS, GAIN_METHOD)


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
