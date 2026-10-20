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

MERGING. Every foreign stack merges — "people may wish to merge masters they
no longer have the raw subframes for". The pipeline stamp
(`hashing.pipeline_stamp`) only LABELS: a foreign master whose stamp matches
one of this instance's own post-processed masters merges silently; one with
another stamp, or none (stacked outside AstroFlow), merges with a CONCERN on
the stack naming it, which the target and mosaic pages bubble up. A master
with no stamp is a raw integration: it goes through this instance's own
post-stack calibration first (`foreign_post_processed_master`), so what the
mosaic merges is on the same flux scale and background as everything else.

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
    pipeline_stamp: str = ""      # the master's PIPEHASH; "" = stacked outside AstroFlow
    same_pipeline: bool = False   # the stamp matches a master this instance built
    observer: str | None = None   # OBSERVER, for credit — never for merging

    @classmethod
    def from_master(cls, meta: ImageMetadata, *, same_pipeline: bool, shared_dir: str) -> "ForeignStack":
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
            # The label is identity: a stamp that starts matching re-keys the
            # targets and mosaics the stack is in, so the concern is re-judged.
            content_hash=hash_list([content_hash, "same" if same_pipeline else "other"]),
            # An AstroFlow master names its oldest subframe; any other master
            # is only its pixels.
            stable_hash=str(h.get("STBLHASH") or meta.stable_hash),
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
            same_pipeline=same_pipeline,
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
    whether our pipeline made it: that one wins, then the smaller hash."""
    from hashing.pipeline_stamp import stamp_of
    chosen: dict[str, tuple] = {}
    for h in hashes:
        meta = masters[h]
        same = stamp_of(meta.header) in local_stamps
        key = str(meta.header.get("STBLHASH") or meta.stable_hash)
        rank = (not same, meta.content_hash)
        if key not in chosen or rank < chosen[key][0]:
            chosen[key] = (rank, meta, same)
    return [(meta, same) for _key, (_rank, meta, same) in sorted(chosen.items())]


@node(resources={"diskcache": 0.001})
def identify_foreign_stacks(
    shared_ref: ContextRef, hashes: tuple, local_stamps: tuple, shared_dirs: tuple,
) -> StacksDict:
    """The foreign stacks for the shared masters `hashes` (keys into
    `shared_ref`), one per stable hash, labelled by whether the master's
    pipeline stamp is one of `local_stamps`."""
    out = StacksDict()
    n_same = 0
    for meta, same in choose_foreign_masters(shared_ref.get(), hashes, local_stamps):
        try:
            stack = ForeignStack.from_master(
                meta, same_pipeline=same,
                shared_dir=shared_dir_of(meta.filename, shared_dirs))
        except Exception as e:
            log.warning("shared master not admissible as a stack",
                        file=str(meta.filename), error=str(e))
            continue
        n_same += same
        out[stack.stable_hash] = ContextRef.store(("stacks", stack.stable_hash), stack)
    log.info("identified foreign stacks", candidates=len(hashes), stacks=len(out),
             same_pipeline=n_same)
    return out


@node(resources={"diskcache": 0.001}, reports_concerns=True)
def foreign_master(stack: ForeignStack) -> ImageMetadata:
    """A foreign stack's master, as the pipelines expect one: the file on
    disk, re-identified under this op's call hash exactly as
    `pipelines.subframe.subframe_channel` does for a raw frame (the runtime
    stamps the blank `content_hash` with the call hash, keeping the
    pure-Merkle handle/value equality every trace prediction depends on).
    Files the concern that labels a master our pipeline did not make."""
    import concerns
    from metadata.image_metadata import compute_image_metadata
    if not stack.pipeline_stamp:
        concerns.warn(
            "Stacked outside AstroFlow",
            f"{Path(stack.master_path).name} carries no pipeline stamp, so it was integrated by "
            "another program. It is calibrated here (Gaia flux scale, background, pedestal) and "
            "merged; its integration and rejection are whatever that program did.",
            observer=stack.observer)
    elif not stack.same_pipeline:
        concerns.warn(
            "Made by a different pipeline",
            f"{Path(stack.master_path).name} was post-processed by an AstroFlow pipeline that is "
            "not one this instance runs (another stacker, or another version of the code). It is "
            "merged as it is.",
            observer=stack.observer, pipeline_stamp=stack.pipeline_stamp)
    meta = compute_image_metadata._function(Path(stack.master_path))
    return dataclasses.replace(meta, content_hash="")


def foreign_post_processed_master(stack: ForeignStack, config, post_process):
    """What a pipeline's per-stack recipe returns for a foreign stack: an
    AstroFlow master as it is (its pipeline post-processed it), any other
    master through THIS pipeline's post-stack calibration `post_process(stack,
    config, master)` — a raw integration from another program has neither
    the Gaia flux scale nor the background treatment the merge assumes."""
    from metadata.solve_nodes import inherit_plate_solution
    master = foreign_master(stack)
    if stack.pipeline_stamp:
        return master
    return inherit_plate_solution(post_process(stack, config, master), master)


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
) -> tuple[dict[int, list[str]], list[list[str]]]:
    """Which spatial group of raw frames each shared master points at — by
    the pointing test `cluster_by_image_center` groups frames with
    (`clustering.spatial.pointing` / `same_pointing`) — so a foreign panel
    joins the local stacks of its pointing in ONE `identify_targets` call and
    lands in the same Target. Masters matching no local pointing are grouped
    among themselves by the same test, one group per pointing:
    `identify_targets` separates by sensor, scale and rotation, never by
    position, so two panes handed to it together would become one Target.
    Returns ({group index: [content hashes]}, [[content hashes], ...] of the
    pointings of their own)."""
    from clustering.spatial import find_groups, pointing, same_pointing

    group_pointings = [
        [pointing(metadata, h, overlap_fraction)
         for h in g if h in metadata and metadata[h].plate_solution is not None]
        for g in spatial_groups
    ]
    by_group: dict[int, list[str]] = {}
    own: dict[str, tuple] = {}
    for h, meta in sorted(masters.items()):
        if meta.plate_solution is None:
            continue
        mine = pointing(MetadataDict({h: meta}), h, overlap_fraction)
        matches = [i for i, frames in enumerate(group_pointings)
                   if any(same_pointing(mine, f) for f in frames)]
        if matches:
            by_group.setdefault(matches[0], []).append(h)
        else:
            own[h] = mine
    own_groups = find_groups(own, same_pointing, 1)
    return by_group, [sorted(g) for g in sorted(own_groups, key=lambda g: sorted(g)[0])]
