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
from decorators import node
from dependency_graph import Graph
from file_io.generic import read_header
from file_io.sidecar import get_section, set_section
from file_io.types import WithSidecar, XisfFilePath
from hashing import hash_list, hash_list_int
from metadata.classes import FrameHash, MetadataDict
from metadata.image_types import is_light_frame, is_master
from metadata.star_photometry import SCHEMA as PHOTOMETRY_SCHEMA
from stacks.composite_stack import CalibrationStack, CalibrationStacksDict, _camera_id_hash_from
from stacks.run import RunsDict
from stacks.starflat import (
    gather_star_photometry, response_image, solve_star_flat, split_half, write_master_flat,
)
from stacks.types import StackHash
from units import Celsius, Seconds

log = structlog.get_logger()

MIN_SUBS = 25                 # fewer solved lights than this: no candidate at all
MIN_SPLIT_HALF = 0.8          # odd/even-sub map agreement below this: built but not offered
CELL_UNCERTAINTY_MAX = 0.0015 # log units; the finest cell whose per-cell noise stays under 0.15%
CELLS = (256, 128, 64)        # 64 px won on held-out stars but printed its noise into masters
                              # at 0.24%/cell (2026-09-07) — the uncertainty rule decides
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
    (`same_optical_train`), greedily along the sorted values."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    clusters: list[list[int]] = []
    for i in order:
        if clusters and values[i] <= 1.02 * values[clusters[-1][0]]:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    return clusters


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


def solve_adaptive(stars: np.ndarray, shape: tuple[int, int]) -> tuple:
    """Solve at CELLS from coarse to fine and keep the finest cell whose
    median per-cell uncertainty stays under CELL_UNCERTAINTY_MAX.
    Returns (solution, cell, r_raw, r_blob)."""
    best = None
    for cell in CELLS:
        sol = solve_star_flat(stars, shape, cell=cell, mag_window=MAG_WINDOW)
        unc = float(np.nanmedian(sol.uncertainty)) if np.isfinite(sol.uncertainty).any() else float("inf")
        if best is None or unc <= CELL_UNCERTAINTY_MAX:
            best = (sol, cell)
        if unc > CELL_UNCERTAINTY_MAX:
            break
    sol, cell = best
    r_raw, r_blob = split_half(stars, sol)
    return sol, cell, r_raw, r_blob


def solve_and_write(star_flat: StarFlatStack, dest) -> dict:
    """Solve the group, write `dest` as a WBPP-ready Master Flat XISF
    carrying the star flat's STBLHASH, and record the quality metrics in
    its sidecar (`SECTION`) for the gate. Returns the metrics."""
    stars = gather_star_photometry([Path(p) for p in star_flat.member_files])
    n_with = int(len(np.unique(stars[:, 10]))) if len(stars) else 0
    shape = (star_flat.height_pixels, star_flat.width_pixels)
    log.info("building star flat", stack=str(star_flat), subs_with_photometry=n_with, stars=len(stars))
    if n_with < MIN_SUBS:
        response = np.ones(shape, np.float32)
        quality = {"n_subs": n_with, "good": False, "reason": "too few subs with photometry"}
    else:
        sol, cell, r_raw, r_blob = solve_adaptive(stars, shape)
        response = response_image(sol, smooth_sigma_cells=0.7)
        quality = {"n_subs": len(sol.kept_subs), "cell": cell, "split_half_r": r_raw, "split_half_blob": r_blob,
                   "per_cell_uncertainty": float(np.nanmedian(sol.uncertainty)), "colour_term": sol.colour_term,
                   "residual_scatter": sol.residual_scatter, "good": bool(np.isfinite(r_raw) and r_raw >= MIN_SPLIT_HALF)}
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    write_master_flat(response, dest, {
        "IMAGETYP": "Master Flat", "FILTER": star_flat.filter or "", "INSTRUME": star_flat.camera,
        "FOCALLEN": star_flat.focal_length, "XBINNING": 1, "YBINNING": 1, "EXPOSURE": 1.0, "EXPTIME": 1.0,
        "STBLHASH": star_flat.stable_hash, "SFLATSUB": quality.get("n_subs", 0), "SFLATCEL": quality.get("cell", 0),
    })
    set_section(Path(dest), SECTION, quality)
    log.info("star flat built", stack=str(star_flat), **quality)
    return quality


@node(num_cpus=4)
def build_star_flat(star_flat: StarFlatStack) -> WithSidecar[XisfFilePath]:
    """The group's master flat (see `solve_and_write`)."""
    dest = build_star_flat.filename_fn(star_flat)
    solve_and_write(star_flat, dest)
    return WithSidecar(XisfFilePath(dest))


def gate(candidates: StarFlatCandidates, master_paths) -> StarFlats:
    """The candidates whose built master passed the gate."""
    out = StarFlats()
    for path in master_paths:
        sid = StackHash(str(read_header(Path(path))["STBLHASH"]))
        q = get_section(Path(path), SECTION) or {}
        if sid in candidates and q.get("good"):
            out[sid] = candidates[sid]
        else:
            log.info("star flat not offered", stack=sid, quality=q)
    return out


@node
def mint_star_flats(candidates: StarFlatCandidates, *masters: WithSidecar) -> StarFlats:
    return gate(candidates, [m.path for m in masters])


@ray.remote(num_cpus=0.1)
def scatter_star_flats(candidates: StarFlatCandidates) -> Graph:
    """One build per candidate, then the gate. The subgraph's result (the
    StarFlats) flows on to the spatial-group scatter as an ancestor."""
    builds = [(build_star_flat, candidates[k]) for k in sorted(candidates)]
    graph = Graph({b: {} for b in builds})
    graph[(mint_star_flats, candidates)] = builds        # ordered: value-consuming collector
    return graph


@node
def merge_star_flats(star_flats: StarFlats, cal_stacks: CalibrationStacksDict) -> CalibrationStacksDict:
    """Star flats join the calibration stacks before cal-to-cal association
    and every light↔flat resolution."""
    return CalibrationStacksDict({**cal_stacks, **star_flats})


def star_flat_master_path(star_flat: StarFlatStack) -> Path | None:
    """The built master's path when it exists in the cache, else None
    (previews must never trigger a build)."""
    hit, value = build_star_flat.cached_value(star_flat)
    return Path(value.path) if hit and value is not None else None
