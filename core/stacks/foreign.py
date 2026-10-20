"""Foreign stacks: another user's master, admitted as a Stack with no
subframes here.

A master found in a shared master folder (`metadata.shared_masters`) becomes
a `ForeignStack`: every attribute a Target or a mosaic reads (filter,
exposure, integration, dates, sensor, centre, scale, rotation, footprint)
comes from the master's own header and sidecar, and its `contents` are
empty. It joins target identification like any stack, so two users pushing
overlapping panels get the same mosaic on both machines without anyone
doing anything. The pipelines take its master from disk (`foreign_master`)
where they would stack and calibrate a local one.

ADMISSION. A foreign stack is MERGED only when the same pipeline made its
master (`hashing.pipeline_stamp`): its `PIPEHASH` must equal the stamp of a
post-processed master this instance built itself. A foreign stack whose
stamp matches nothing local is still identified — it shows on the starmap,
dashed and grey, with the reason — but no pipeline merges it
(`targets.target_channel.group_stacks_by_filter`). Admission is part of the
stack's `content_hash`, so a stamp that starts matching (this instance built
its first master, or the other side updated and re-pushed) re-keys the
target and the mosaic, and the pipelines run again with the panel in.

IDENTITY. `stable_hash` is the master's `STBLHASH` — its oldest subframe,
stamped by the AstroFlow that stacked it — so a foreign master of the very
subframes a local stack holds (the user's own published copy coming back;
a remote-observatory dataset two members both bought) collides with the local
stack's key and the local one wins (`combine_dicts`, local last). Two
foreign masters of one stable hash (a re-push whose predecessor the sync tool
has not yet deleted) keep the admitted one, then the smaller call hash.
"""
from __future__ import annotations

import dataclasses
import datetime
from dataclasses import dataclass
from pathlib import Path

import structlog
from beartype import beartype
from frozendict import frozendict

from attributes.astrometric_attributes import AstrometricAttributes
from cache.context_ref import ContextRef
from decorators import node
from hashing import hash_list
from metadata.classes import ImageMetadata, MetadataDict
from stacks.composite_stack import Stack, StacksDict
from stacks.run import RunsDict
from units import Celsius, Seconds

log = structlog.get_logger()

#: The sidecar section `stacks.publish_shared` writes beside a published
#: master: what the consumer cannot read from the master itself.
SHARED_SECTION = "shared"


@beartype
@dataclass(frozen=True)
class ForeignStack(Stack):
    """A Stack whose master was made elsewhere; `contents` is empty."""
    master_path: str = ""
    shared_dir: str = ""          # the shared master folder it was found in
    pipeline_stamp: str = ""      # the master's PIPEHASH
    admitted: bool = False        # the stamp matches a locally built master's
    observer: str | None = None   # OBSERVER, for credit — never for admission

    @classmethod
    def from_master(cls, meta: ImageMetadata, *, admitted: bool, shared_dir: str) -> "ForeignStack":
        from file_io.sidecar import get_section
        from hashing.pipeline_stamp import stamp_of
        h = meta.header
        shared = get_section(meta.filename, SHARED_SECTION) or {}
        exposure = Seconds(float(h.get("EXPTIME") or h.get("EXPOSURE") or 0.0))
        integration = Seconds(float(shared.get("integration_s") or meta.master_integration_s or exposure))
        temperature = h.get("CCD-TEMP", h.get("SET-TEMP"))
        content_hash = meta.content_hash
        frames = MetadataDict({content_hash: meta})
        astrometric = AstrometricAttributes.from_frames([content_hash], frames)
        return cls(
            # The admission state is identity: a stamp that starts matching
            # must re-key every target and mosaic the stack belongs to.
            content_hash=hash_list([content_hash, "admitted" if admitted else "pending"]),
            stable_hash=str(h["STBLHASH"]),
            oldest_date=_date(shared.get("oldest_date"), h["DATE-OBS"]),
            newest_date=_date(shared.get("newest_date"), h["DATE-OBS"]),
            image_type="light",
            camera=str(h.get("INSTRUME") or ""),
            filter=(str(h["FILTER"]) if h.get("FILTER") not in (None, "") else None),
            mean_temperature=Celsius(float(temperature) if temperature is not None else 0.0),
            exposure=exposure,
            integration=integration,
            width_pixels=meta.width_pixels,
            height_pixels=meta.height_pixels,
            state=frozendict(),
            contents=RunsDict(),
            camera_id_hash=None,
            **dataclasses.asdict(astrometric),
            master_path=str(meta.filename),
            shared_dir=shared_dir,
            pipeline_stamp=stamp_of(h) or "",
            admitted=admitted,
            observer=(str(shared.get("observer") or h.get("OBSERVER") or "").strip() or None),
        )


def _date(iso: str | None, fallback: datetime.datetime) -> datetime.datetime:
    """UTC-aware, like every local stack's dates (`clustering.quiesce._utc`):
    a naive date from another program's header cannot be compared with them."""
    from clustering.quiesce import _utc
    if iso:
        try:
            return _utc(datetime.datetime.fromisoformat(str(iso)))
        except ValueError:
            pass
    return _utc(fallback)


def is_foreign(stack) -> bool:
    return isinstance(stack, ForeignStack)


def shared_dir_of(path: str | Path, shared_dirs: tuple[str, ...]) -> str:
    """The configured shared folder `path` lies under ('' when none does)."""
    p = Path(path)
    for d in shared_dirs:
        if p.is_relative_to(Path(d)):
            return str(d)
    return ""


def choose_foreign_masters(masters: dict, hashes, local_stamps) -> list[tuple[ImageMetadata, bool]]:
    """One master per stable hash among `hashes` (keys into `masters`), with
    its admission: the admitted one wins, then the smaller call hash."""
    from hashing.pipeline_stamp import stamp_of
    chosen: dict[str, tuple] = {}
    for h in hashes:
        meta = masters[h]
        admitted = stamp_of(meta.header) in local_stamps
        key = str(meta.header["STBLHASH"])
        rank = (not admitted, meta.content_hash)
        if key not in chosen or rank < chosen[key][0]:
            chosen[key] = (rank, meta, admitted)
    return [(meta, admitted) for _key, (_rank, meta, admitted) in sorted(chosen.items())]


@node(resources={"diskcache": 0.001})
def identify_foreign_stacks(
    shared_ref: ContextRef, hashes: tuple, local_stamps: tuple, shared_dirs: tuple,
) -> StacksDict:
    """The foreign stacks for the shared masters `hashes` (keys into
    `shared_ref`), one per stable hash, admitted when the master's pipeline
    stamp is one of `local_stamps`."""
    out = StacksDict()
    n_admitted = 0
    for meta, admitted in choose_foreign_masters(shared_ref.get(), hashes, local_stamps):
        try:
            stack = ForeignStack.from_master(
                meta, admitted=admitted,
                shared_dir=shared_dir_of(meta.filename, shared_dirs))
        except Exception as e:
            log.warning("shared master not admissible as a stack",
                        file=str(meta.filename), error=str(e))
            continue
        n_admitted += admitted
        out[stack.stable_hash] = ContextRef.store(("stacks", stack.stable_hash), stack)
    log.info("identified foreign stacks", candidates=len(hashes), stacks=len(out),
             admitted=n_admitted)
    return out


@node(resources={"diskcache": 0.001})
def foreign_master(stack: ForeignStack) -> ImageMetadata:
    """A foreign stack's post-processed master, as the pipelines expect one:
    the file on disk, re-identified under this op's call hash exactly as
    `pipelines.subframe.subframe_channel` does for a raw frame (the runtime
    stamps the blank `content_hash` with the call hash, keeping the
    pure-Merkle handle/value equality every trace prediction depends on)."""
    from metadata.image_metadata import compute_image_metadata
    meta = compute_image_metadata._function(Path(stack.master_path))
    return dataclasses.replace(meta, content_hash="")


def local_pipeline_stamps() -> tuple[str, ...]:
    """The pipeline stamps of the post-processed masters THIS instance built
    — the reference a foreign master's stamp must match. Read from the
    product version log (every current master, every pipeline) and the
    masters' SIDECARS (the header there is the file's at landing, and it is
    what a consumer reads of a published copy); head-only (the durable
    store), like the orphan sweep."""
    from context.cache import get_durable_cache
    from hashing.pipeline_stamp import stamp_of
    from metadata.sidecar_io import read_image_metadata
    dc = get_durable_cache()
    stamps: set[str] = set()
    for k in list(dc):
        if not (isinstance(k, tuple) and len(k) == 6 and k[0] == "product_versions"
                and k[1] == "Stack" and k[4] == "Post-Processed"):
            continue
        for entry in dc.get(k) or []:
            path = Path(str(entry.get("path", "")))
            if not path.is_file():
                continue
            try:
                meta = read_image_metadata(path)
            except Exception:
                continue
            stamp = stamp_of(meta.header) if meta is not None else None
            if stamp:
                stamps.add(stamp)
    return tuple(sorted(stamps))


def assign_to_spatial_groups(
    masters: dict, metadata: MetadataDict, spatial_groups: list, overlap_fraction: float,
) -> tuple[dict[int, list[str]], list[str]]:
    """Which spatial group of raw frames each shared master points at — the
    same centre-and-field-of-view test `cluster_by_image_center` groups frames
    by — so a foreign panel joins the local stacks of its pointing in ONE
    `identify_targets` call and lands in the same Target. Returns
    ({group index: [content hashes]}, [content hashes matching no group])."""
    import math

    import numpy as np

    from clustering.spatial import _min_fov_deg
    from geometry import footprints

    def threshold(meta_dict, h) -> float:
        max_offset_deg = (1.0 - overlap_fraction) * _min_fov_deg(meta_dict, h) / 2.0
        return math.cos(math.radians(min(max_offset_deg, 2.0)))

    group_vectors = []
    for g in spatial_groups:
        frames = [h for h in g if h in metadata and metadata[h].plate_solution is not None]
        if not frames:
            group_vectors.append((np.zeros((0, 3)), np.zeros(0)))
            continue
        vs = np.array([footprints.radec_to_vector(*metadata[h].plate_solution.center) for h in frames])
        ts = np.array([threshold(metadata, h) for h in frames])
        group_vectors.append((vs, ts))

    by_group: dict[int, list[str]] = {}
    unassigned: list[str] = []
    for h, meta in sorted(masters.items()):
        if meta.plate_solution is None:
            continue
        v = np.asarray(footprints.radec_to_vector(*meta.plate_solution.center))
        t = threshold(MetadataDict({h: meta}), h)
        best, best_dot = None, -2.0
        for i, (vs, ts) in enumerate(group_vectors):
            if not len(vs):
                continue
            dots = vs @ v
            ok = dots > np.maximum(ts, t)
            if ok.any() and dots[ok].max() > best_dot:
                best, best_dot = i, float(dots[ok].max())
        if best is None:
            unassigned.append(h)
        else:
            by_group.setdefault(best, []).append(h)
    return by_group, unassigned
