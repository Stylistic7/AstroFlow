"""Dust maps from the science frames: one per optical system, night and
pointing (a DUST RUN), measured once and cached for ever; folded into mote
epochs per optical system by `stacks/dust_epochs.py`.

Why the subs and not the flats (Stylistic7, 2026-10-13): flats are shot rarely
and the dust moves with every reassembly, so the frames themselves are the
only record of which motes each night saw. A mote is fixed in pixel
coordinates while the sky is not: the temporal median of many flattened,
star-masked subs keeps the dust and loses the sky.

The pedestal is the camera's OFFSET: on both ZWO cameras here a master dark
sits at exactly 10 x OFFSET ADU whatever the exposure (50 -> 501-502 ADU,
measured 2026-10-13 on 30-300 s darks), so no dark master is needed and a
map depends on nothing but its own subs. A percentile pedestal sits above
the bias and inflates every depth ~2x — the rule is a measurement, not a
guess.

Graph: `identify_dust_runs` (one node over the library) -> `scatter_dust_runs`
(a builder: one `measure_dust_run` per run, one `dust_epochs` collector per
system, `merge_dust_epochs`) -> `store_dust_epochs` into the context, where
`calibrate` and the stackers look each stack run up by its frames.
"""
# NO `from __future__ import annotations`: the @node decorator reads the
# return annotation to recognise a file PRODUCER (`WithSidecar[FitsFilePath]`
# → filename_fn + the metadata wrap); as a string it is not one (boot181).
import datetime as _dt
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ray
import structlog
from astropy.io import fits
from frozendict import frozendict
from scipy import ndimage

from cache.content_hashed import ContentHashed
from cache.context_ref import ContextRef
from clustering.quiesce import _utc
from context import context
from decorators import node
from dependency_graph import Graph
from file_io.generic import read_image
from file_io.sidecar import set_section
from file_io.types import FitsFilePath, WithSidecar
from hashing import hash_list, hash_list_int
from metadata.classes import FrameHash, ImageMetadata, MetadataDict
from metadata.image_types import is_light_frame, is_master
from stacks.composite_stack import _camera_id_hash_from
from stacks.dust_epochs import Epoch, MoteModel, RunMap, fold
from stacks.star_flat import _focal_clusters

log = structlog.get_logger()

BLOCK = 8                 # px per block: the map's resolution (the sky flat's too)
FLATTEN_PX = 800.0        # a sub is divided by this broad smooth of itself first
STAR_RATIO = 1.05         # above the smooth by this much is a star or nebula: masked
MIN_SUBS = 3
NIGHT_GAP = _dt.timedelta(hours=8)
PEDESTAL_PER_OFFSET = 10.0
SECTION = "dust_run"


@dataclass(frozen=True, kw_only=True)
class DustRun(ContentHashed):
    """The lights of one optical system on one night at one pointing.
    Identity is the frames (`content_hash`); the paths ride along to be read."""
    system: str
    frames: tuple[str, ...]
    files: tuple[str, ...]
    date: _dt.datetime
    centre: tuple[float, float]
    width_pixels: int
    height_pixels: int

    def __hash__(self) -> int:
        return hash_list_int([self.content_hash])


class DustRuns(dict[str, DustRun]):
    def __hash__(self):
        return hash_list_int([self[k].content_hash for k in sorted(self)])


@dataclass(frozen=True)
class DustEpochs:
    """Every dust run's model, by run and by frame, for all optical systems."""
    model_of_run: frozendict
    run_of_frame: frozendict
    report: tuple
    content_hash: str = ""

    @classmethod
    def empty(cls) -> "DustEpochs":
        return cls(frozendict(), frozendict(), (), "none")

    @classmethod
    def of(cls, model_of_run: dict, run_of_frame: dict, report: tuple) -> "DustEpochs":
        identity = hash_list([(r, model_of_run[r].content_hash) for r in sorted(model_of_run)])
        return cls(frozendict(model_of_run), frozendict(run_of_frame), report, identity)

    def model_for(self, frames) -> MoteModel | None:
        """The model of the dust run most of `frames` belong to, or None."""
        runs = [self.run_of_frame[f] for f in frames if f in self.run_of_frame]
        if not runs:
            return None
        return self.model_of_run[max(set(runs), key=runs.count)]


# ------------------------------------------------------------------ identify
def _system_of(m: ImageMetadata, fingerprints: dict, h: str) -> str | None:
    header = m.header
    if not is_light_frame(header) or is_master(header) or m.plate_solution is None:
        return None
    try:
        focal = float(header.get("FOCALLEN") or 0)
    except (TypeError, ValueError):
        return None
    if focal <= 0:
        return None
    return _camera_id_hash_from(m, fingerprints.get(FrameHash(h), ())) or f"{m.width_pixels}x{m.height_pixels}"


def dust_runs(metadata: MetadataDict, fingerprints: dict) -> DustRuns:
    """Lights grouped by optical system (camera identity x focal-length
    cluster, as star flats are), then by night (an 8 h gap in the capture
    times), then by pointing (`assign_pointings`' separation)."""
    from stacks.dust_epochs import POINTING_SEP_DEG
    by_cam: dict[str, list] = defaultdict(list)
    for h, m in metadata.items():
        cam = _system_of(m, fingerprints, h)
        if cam is not None:
            by_cam[cam].append((float(m.header["FOCALLEN"]), h, m))
    out = DustRuns()
    for cam, items in by_cam.items():
        for idx in _focal_clusters([f for f, _, _ in items]):
            members = sorted((items[i] for i in idx), key=lambda t: (_utc(t[2].header["DATE-OBS"]), t[1]))
            system = f"{cam[:8]}@{int(round(np.median([f for f, _, _ in members])))}"
            nights: list[list] = []
            for f, h, m in members:
                if nights and _utc(m.header["DATE-OBS"]) - _utc(nights[-1][-1][2].header["DATE-OBS"]) <= NIGHT_GAP:
                    nights[-1].append((f, h, m))
                else:
                    nights.append([(f, h, m)])
            for night in nights:
                pointings: list[list] = []
                for f, h, m in night:
                    ra, dec = m.plate_solution.center
                    for p in pointings:
                        ra0, dec0 = p[0][2].plate_solution.center
                        dra = (ra - ra0 + 180.0) % 360.0 - 180.0
                        if np.hypot(dra * np.cos(np.radians(dec0)), dec - dec0) < POINTING_SEP_DEG:
                            p.append((f, h, m))
                            break
                    else:
                        pointings.append([(f, h, m)])
                for p in pointings:
                    if len(p) < MIN_SUBS:
                        continue
                    frames = tuple(sorted(h for _, h, _ in p))
                    first = p[0][2]
                    run = DustRun(
                        content_hash=hash_list(list(frames)), system=system, frames=frames,
                        files=tuple(str(m.filename) for _, _, m in p), date=_utc(first.header["DATE-OBS"]),
                        centre=(float(np.mean([m.plate_solution.center[0] for _, _, m in p])),
                                float(np.mean([m.plate_solution.center[1] for _, _, m in p]))),
                        width_pixels=first.width_pixels, height_pixels=first.height_pixels)
                    out[run.content_hash] = run
    return out


@node
def identify_dust_runs(metadata_ref: ContextRef[MetadataDict], fingerprint_map_ref: ContextRef) -> DustRuns:
    runs = dust_runs(metadata_ref.get(), fingerprint_map_ref.get())
    systems = defaultdict(int)
    for r in runs.values():
        systems[r.system] += 1
    log.info("dust runs identified", runs=len(runs), subs=sum(len(r.frames) for r in runs.values()),
             systems=dict(systems))
    return runs


# ------------------------------------------------------------------ measure
def block_mean(a: np.ndarray, k: int) -> np.ndarray:
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3), dtype=np.float64).astype(np.float32)


def flattened_sub(sub: np.ndarray, pedestal: float, block: int = BLOCK) -> np.ndarray:
    """One sub's block image divided by its own broad smooth, stars and
    nebulosity (above the smooth) masked to NaN."""
    x = block_mean(np.asarray(sub, np.float32) - pedestal, block)
    smooth = ndimage.gaussian_filter(x, FLATTEN_PX / block)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = x / smooth
    r[(x <= 0) | (smooth <= 0) | (r > STAR_RATIO)] = np.nan
    return r


def combine_dust_map(files, block: int = BLOCK) -> tuple[np.ndarray, int]:
    maps = []
    for f in files:
        header = fits.getheader(f)
        sub = np.asarray(read_image(Path(f)), np.float32).squeeze()
        maps.append(flattened_sub(sub, PEDESTAL_PER_OFFSET * float(header.get("OFFSET", 0) or 0), block))
    out = np.nanmedian(np.stack(maps), axis=0)
    return np.nan_to_num(out, nan=1.0).astype(np.float32), len(maps)


@node(num_cpus=1, resources={"diskcache": 0.001})
def measure_dust_run(run: DustRun) -> WithSidecar[FitsFilePath]:
    """The dust map of one run: the temporal median of its flattened,
    star-masked subs at BLOCK resolution. Head-pinned like every reader of
    raw subs (a joining worker briefly reports existing files as missing)."""
    missing = [f for f in run.files if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(run.files)} subs unreadable here, e.g. {missing[0]}")
    dest = Path(measure_dust_run.filename_fn(run))
    dest.parent.mkdir(parents=True, exist_ok=True)
    out, n = combine_dust_map(run.files)
    hdu = fits.PrimaryHDU(data=out)
    hdu.header["IMAGETYP"] = "Dust Map"
    hdu.header["DUSTSYS"] = run.system
    hdu.header["DUSTRUN"] = run.content_hash
    hdu.header["DATE-OBS"] = run.date.isoformat()
    hdu.header["NSUBS"] = n
    hdu.header["BLOCK"] = BLOCK
    hdu.header["DUSTRA"] = run.centre[0]
    hdu.header["DUSTDEC"] = run.centre[1]
    hdu.writeto(dest, overwrite=True)
    quality = {"system": run.system, "n_subs": n, "date": run.date.isoformat(), "centre": list(run.centre),
               "block": BLOCK, "flat_rms": round(float(np.std(out)), 5)}
    set_section(dest, SECTION, quality)
    log.info("dust map measured", run=run.content_hash[:8], **quality)
    return WithSidecar(FitsFilePath(dest))


# ------------------------------------------------------------------ fold
def _report(system: str, runs: tuple[DustRun, ...], epochs: list[Epoch]) -> dict:
    return {"system": system, "runs": len(runs),
            "epochs": [{"from": runs[e.start].date.isoformat(), "runs": len(e.runs),
                        "model": e.model.summary()} for e in epochs]}


@node(num_cpus=2, resources={"diskcache": 0.001})
def dust_epochs(system: str, runs: tuple, *maps: ImageMetadata) -> DustEpochs:
    """The epochs of one optical system from its runs' maps (`runs` and
    `maps` in the same date order). A pure fold, recomputed whole on every
    pass: a closed epoch's model comes out identical whatever follows it."""
    assert len(runs) == len(maps), (len(runs), len(maps))
    run_maps = [RunMap(run=r.content_hash, date=r.date, centre=r.centre, n_subs=len(r.frames),
                       map=np.asarray(read_image(Path(m.filename)), np.float32).squeeze())
                for r, m in zip(runs, maps)]
    epochs = fold(system, run_maps, BLOCK)
    model_of_run = {runs[i].content_hash: e.model for e in epochs for i in e.runs}
    run_of_frame = {f: r.content_hash for r in runs for f in r.frames}
    report = _report(system, runs, epochs)
    log.info("dust epochs folded", **report)
    return DustEpochs.of(model_of_run, run_of_frame, (report,))


@node
def merge_dust_epochs(*tables: DustEpochs) -> DustEpochs:
    model_of_run, run_of_frame, report = {}, {}, ()
    for t in tables:
        model_of_run.update(t.model_of_run)
        run_of_frame.update(t.run_of_frame)
        report += t.report
    return DustEpochs.of(model_of_run, run_of_frame, report)


@ray.remote(num_cpus=0.1)
def scatter_dust_runs(runs: DustRuns) -> Graph:
    """One measurement per dust run, one fold per optical system (its runs
    in date order, the value-consuming collector's deps in the same order),
    merged into one table."""
    by_system: dict[str, list[DustRun]] = defaultdict(list)
    for r in runs.values():
        by_system[r.system].append(r)
    graph = Graph({})
    collectors = []
    for system in sorted(by_system):
        ordered = tuple(sorted(by_system[system], key=lambda r: (r.date, r.content_hash)))
        measures = [(measure_dust_run, r) for r in ordered]
        for m in measures:
            graph[m] = {}
        collector = (dust_epochs, system, ordered)
        graph[collector] = measures
        collectors.append(collector)
    graph[merge_dust_epochs] = collectors
    return graph


@ray.remote(num_cpus=0.1)
def store_dust_epochs(table: DustEpochs) -> DustEpochs:
    context.dust_epochs = table
    log.info("Stored dust epochs", runs=len(table.model_of_run),
             epochs=sum(len(r["epochs"]) for r in table.report))
    return table
