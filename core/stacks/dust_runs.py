"""Dust maps from the science frames: one per RUN (a stack's run — one
night, one pointing, one filter), measured once and cached for ever, folded
into mote epochs per RIG and filter by `stacks/dust_epochs.py`. A rig is a
camera (its defect-fingerprint identity, as every stack uses) at a focal
length (clustered within 2 %, as star flats do).

Why the subs and not the flats (Stylistic7, 2026-10-13): flats are shot rarely
and the dust moves with every reassembly, so the frames themselves are the
only record of which motes each night saw. A mote is fixed in pixel
coordinates while the sky is not: the temporal median of many flattened,
star-masked subs keeps the dust and loses the sky.

Per FILTER (Stylistic7): dust on the sensor window is common to every filter,
dust on a filter is that filter's alone, and a map pooled over filters would
hold a filter mote in a minority of its subs and lose it in the median. So
each rig and filter folds its own epochs — a sensor change shows as a
boundary in every filter's fold, a filter change in one.

The pedestal is the camera's OFFSET: on both ZWO cameras here a master dark
sits at exactly 10 x OFFSET ADU whatever the exposure (50 -> 501-502 ADU,
measured 2026-10-13 on 30-300 s darks), so no dark master is needed and a
map depends on nothing but its own subs. A percentile pedestal sits above
the bias and inflates every depth ~2x — the rule is a measurement, not a
guess.

Graph: `scatter_dust_runs` (a builder over the identified stacks: one
`measure_dust_run` per run, one `dust_epochs` collector per rig and filter,
`merge_dust_epochs`) -> `store_dust_epochs` into the context, where
`calibrate` and the stackers look each stack run up by its frames.
"""
# NO `from __future__ import annotations`: the @node decorator reads the
# return annotation to recognise a file PRODUCER (`WithSidecar[FitsFilePath]`
# → filename_fn + the metadata wrap); as a string it is not one (boot181).
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import ray
import structlog
from astropy.io import fits
from frozendict import frozendict
from scipy import ndimage

from clustering.quiesce import _utc
from context import context
from decorators import node
from dependency_graph import Graph
from file_io.generic import read_image
from file_io.sidecar import set_section
from file_io.types import FitsFilePath, WithSidecar
from hashing import hash_list
from metadata.classes import ImageMetadata
from stacks.dust_epochs import Epoch, MoteModel, RunMap, fold
from stacks.run import Run
from stacks.star_flat import _focal_clusters

log = structlog.get_logger()

BLOCK = 8                 # px per block: the map's resolution (the sky flat's too)
FLATTEN_PX = 800.0        # a sub is divided by this broad smooth of itself first
STAR_RATIO = 1.05         # above the smooth by this much is a star or nebula: masked
PEDESTAL_PER_OFFSET = 10.0
SECTION = "dust_run"
# the header normaliser admits only bias/dark/flat/light wordings (boot186:
# "Dust Map" broke every map in the metadata wrap); a dust map IS a flat-like
# multiplicative map of the lights' shadows
IMAGE_TYPE = "Dust Flat"


@dataclass(frozen=True)
class DustEpochs:
    """Every run's model, by run and by frame, for all rigs and filters;
    `report` is the fold's record per rig and filter (the UI's timeline)."""
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
        """The model of the run most of `frames` belong to, or None."""
        runs = [self.run_of_frame[f] for f in frames if f in self.run_of_frame]
        if not runs:
            return None
        return self.model_of_run[max(set(runs), key=runs.count)]


# ------------------------------------------------------------------ rigs
@dataclass(frozen=True)
class Rig:
    """A camera (its identity's first 8 hex) at a focal length, with its
    sensor size. Hashable: it rides in the fold's task key."""
    camera_id: str
    camera: str
    focal_length: int
    width: int
    height: int


def _focal_length_of(stack) -> float | None:
    values = []
    for run in stack.contents.values():
        for m in run.metadata.values():
            try:
                values.append(float(m.header.get("FOCALLEN") or 0))
            except (TypeError, ValueError):
                pass
    values = [v for v in values if v > 0]
    return float(np.median(values)) if values else None


def rigs(stacks) -> dict[str, dict]:
    """The light stacks' runs by rig and filter: `{system: {"rig": {...},
    "filter": f, "runs": [Run, ...]}}` with `system` = `<camera>@<focal>/<filter>`.
    A rig is the camera identity (defect fingerprint, else sensor geometry)
    at a focal-length cluster (2 %, single-link — one 350 mm rig reports
    346-352)."""
    by_cam: dict[str, list] = defaultdict(list)
    for stack in stacks:
        if getattr(stack, "image_type", "light") != "light":
            continue
        focal = _focal_length_of(stack)
        if focal is None:
            continue
        cam = stack.camera_id_hash or f"{stack.width_pixels}x{stack.height_pixels}"
        by_cam[cam].append((focal, stack))
    out: dict[str, dict] = {}
    for cam, items in by_cam.items():
        for idx in _focal_clusters([f for f, _ in items]):
            focal = int(round(float(np.median([items[i][0] for i in idx]))))
            for i in idx:
                stack = items[i][1]
                filt = str(stack.filter or "").strip()
                system = f"{cam[:8]}@{focal}/{filt}"
                entry = out.setdefault(system, {
                    "rig": Rig(cam[:8], stack.camera, focal, stack.width_pixels, stack.height_pixels),
                    "filter": filt, "runs": []})
                entry["runs"].extend(stack.contents.values())
    return out


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
def measure_dust_run(run: Run) -> WithSidecar[FitsFilePath]:
    """The dust map of one run: the temporal median of its flattened,
    star-masked subs at BLOCK resolution. Head-pinned like every reader of
    raw subs (a joining worker briefly reports existing files as missing)."""
    files = [str(m.filename) for m in run.metadata.values()]
    missing = [f for f in files if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(files)} subs unreadable here, e.g. {missing[0]}")
    dest = Path(measure_dust_run.filename_fn(run))
    dest.parent.mkdir(parents=True, exist_ok=True)
    out, n = combine_dust_map(files)
    hdu = fits.PrimaryHDU(data=out)
    hdu.header["IMAGETYP"] = IMAGE_TYPE
    hdu.header["DUSTRUN"] = run.content_hash
    hdu.header["DATE-OBS"] = _utc(run.oldest_date).isoformat()
    hdu.header["NSUBS"] = n
    hdu.header["BLOCK"] = BLOCK
    hdu.header["DUSTRA"] = float(run.center[0])
    hdu.header["DUSTDEC"] = float(run.center[1])
    hdu.writeto(dest, overwrite=True)
    quality = {"n_subs": n, "date": _utc(run.oldest_date).isoformat(), "centre": [float(c) for c in run.center],
               "block": BLOCK, "flat_rms": round(float(np.std(out)), 5)}
    set_section(dest, SECTION, quality)
    log.info("dust map measured", run=run.content_hash[:8], **quality)
    return WithSidecar(FitsFilePath(dest))


# ------------------------------------------------------------------ fold
def _report(system: str, rig: Rig, filt: str, runs: tuple, epochs: list[Epoch]) -> dict:
    return {"system": system, "rig": asdict(rig), "filter": filt, "runs": len(runs),
            "epochs": [{"from": _utc(runs[e.runs[0]].oldest_date).isoformat(),
                        "to": _utc(runs[e.runs[-1]].newest_date).isoformat(),
                        "runs": len(e.runs), "pointings": e.pointings, "model": e.model.summary()}
                       for e in epochs]}


@node(num_cpus=2, resources={"diskcache": 0.001})
def dust_epochs(system: str, rig: Rig, filt: str, runs: tuple, *maps: ImageMetadata) -> DustEpochs:
    """The epochs of one rig and filter from its runs' maps (`runs` and
    `maps` in the same date order). A pure fold, recomputed whole on every
    pass: a closed epoch's model comes out identical whatever follows it."""
    assert len(runs) == len(maps), (len(runs), len(maps))
    run_maps = [RunMap(run=r.content_hash, date=_utc(r.oldest_date), centre=tuple(float(c) for c in r.center),
                       n_subs=len(r.frames), map=np.asarray(read_image(Path(m.filename)), np.float32).squeeze())
                for r, m in zip(runs, maps)]
    epochs = fold(system, run_maps, BLOCK)
    model_of_run = {runs[i].content_hash: e.model for e in epochs for i in e.runs}
    run_of_frame = {f: r.content_hash for r in runs for f in r.frames}
    report = _report(system, rig, filt, runs, epochs)
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


def dust_graph(stacks) -> Graph:
    """One measurement per run, one fold per rig and filter (its runs in
    date order, the value-consuming collector's deps in the same order),
    merged into one table."""
    graph = Graph({})
    collectors = []
    groups = rigs(stacks)
    for system in sorted(groups):
        g = groups[system]
        ordered = tuple(sorted(g["runs"], key=lambda r: (_utc(r.oldest_date), r.content_hash)))
        measures = [(measure_dust_run, r) for r in ordered]
        for m in measures:
            graph[m] = {}
        collector = (dust_epochs, system, g["rig"], g["filter"], ordered)
        graph[collector] = measures
        collectors.append(collector)
    log.info("dust runs scattered", systems=len(groups), runs=sum(len(g["runs"]) for g in groups.values()))
    graph[merge_dust_epochs] = collectors
    return graph


@ray.remote(num_cpus=0.1)
def scatter_dust_runs(stacks: dict) -> Graph:
    """`stacks` is the identified light stacks, their values ContextRefs."""
    return dust_graph([v.get() if hasattr(v, "get") else v for v in stacks.values()])


@ray.remote(num_cpus=0.1)
def store_dust_epochs(table: DustEpochs) -> DustEpochs:
    context.dust_epochs = table
    log.info("Stored dust epochs", runs=len(table.model_of_run),
             epochs=sum(len(r["epochs"]) for r in table.report))
    return table
