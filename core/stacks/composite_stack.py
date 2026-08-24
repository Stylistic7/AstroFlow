from dataclasses import dataclass, asdict
from typing import Any, Optional
from collections.abc import Sequence
import ray
from beartype import beartype
from frozendict import frozendict

import units
from hashing import hash_list, hash_list_int
from stacks.run import BaseRun, Run, RunsDict, run_from_frames
from attributes import (
    AstrometricAttributes,
    AstroFlowStateAttributes,
    ChronologyAttributes,
    ImagingAttributes,
)
from stacks.types import StackHash
from metadata.classes import MetadataDict, FrameHash
from clustering.continuous import cluster_continuous
from cache.content_hashed import ContentHashed
from cache.context_ref import ContextRef


def _camera_id_hash_from(
    imaging: "ImagingAttributes", consensus: tuple[int, ...]
) -> str | None:
    """Stable hash binding camera identity to (sensor shape, defect consensus).
    None if consensus is empty (no fingerprint → no known camera).

    Derived from PIXELS ONLY. `INSTRUME` is deliberately absent: it is free
    text the capture software chooses, it is not unique to a body (every
    ASI2600MM Duo ships the same string), and it can change under the user
    mid-session — on 2026-02-16 one camera renamed itself from
    "ASI Camera (1)" to "ZWO ASI2600MM Duo" when its driver reconnected, which
    split one physical sensor into two identities and cost four Starbase light
    stacks their flat masters. A defect consensus is a property of the silicon
    and survives a driver's opinion of itself.

    Sensor shape stays in the hash for two reasons, and it is not a header
    string: consensus entries are flat pixel INDICES, which only mean anything
    relative to a known array shape, and shape separates two sensors that
    improbably agree on a consensus tuple.
    """
    if not consensus:
        return None
    return hash_list([
        (imaging.height_pixels, imaging.width_pixels),
        consensus,
    ])


@beartype
@dataclass(frozen=True)
class BaseStack(
    ContentHashed,
    ImagingAttributes,
    ChronologyAttributes[StackHash],
    AstroFlowStateAttributes,
):
    """Common base for all stacks (light and calibration). Contains Runs."""
    contents: RunsDict
    # Hash of the canonical defect-fingerprint consensus for this Stack's
    # camera. None for bias-only calibration stacks (no defect signal) and
    # for any other stack whose frames lack a confident camera identity.
    camera_id_hash: str | None = None

    def all_frames(self) -> list[FrameHash]:
        all_frames = []
        for c in self.contents.values():
            all_frames.extend(c.frames)
        return all_frames

    def all_metadata(self) -> MetadataDict:
        all_metadata = MetadataDict({})
        for c in self.contents.values():
            all_metadata |= c.metadata
        return all_metadata

    def __hash__(self):
        assert self.content_hash, f"{type(self).__name__} has empty content_hash"
        return hash_list_int([self.content_hash])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseStack):
            return NotImplemented
        return (self.content_hash == other.content_hash) and (
            self.stable_hash == other.stable_hash
        )


@beartype
@dataclass(frozen=True)
class Stack(BaseStack, AstrometricAttributes):
    """Light stack with astrometric attributes (center, footprint, etc.)."""

    @classmethod
    def from_frames(
        cls,
        frames: Sequence[FrameHash],
        metadata: MetadataDict,
        max_run_gap,
        camera_consensus: tuple[int, ...] = (),
    ) -> Optional["Stack"]:
        metadata_list = [i for h, i in metadata.items() if h in frames]

        # Sort the light frames by observation date. The oldest image will be used
        # to compute the hash of the whole stack.
        sorted_metadata = sorted(metadata_list, key=lambda i: i.header["DATE-OBS"])

        # Stack identification clusters frames by exposure before calling
        # us, so every frame here shares an EXPOSURE. Assert that
        # invariant — a stale identification path would produce a Stack
        # that breaks WBPP downstream.
        exposures = {
            i.header["EXPOSURE"]
            for i in metadata_list
            if "EXPOSURE" in i.header
        }
        assert len(exposures) <= 1, (
            f"Stack.from_frames expected one EXPOSURE, got {sorted(exposures)}"
        )

        contents = RunsDict()
        runs = cluster_continuous(
            [
                (FrameHash(i.content_hash), i.header["DATE-OBS"])
                for i in sorted_metadata
            ],
            epsilon=max_run_gap,
        )

        for run in runs:
            run_frames = [x[0] for x in run]
            new_stack = ray.get(
                run_from_frames.remote(run_frames, metadata)
            )
            contents[new_stack.stable_hash] = new_stack

        if not contents:
            return None

        chronology_attributes = ChronologyAttributes.from_frames(frames, metadata)
        imaging_attributes = ImagingAttributes.from_frames(frames, metadata)
        # A composite's runs are all the same filter and pointing, so there
        # is no reference CHANNEL to pick — frame on the oldest run, matching
        # the convention `stable_hash` already uses. The footprint stays the
        # INTERSECTION: WBPP crops the integration to where the subframes
        # overlap, so that is the master's real extent.
        from attributes.astrometric_attributes import intersect_polygons
        runs_by_age = sorted(contents.values(), key=lambda r: r.oldest_date)
        astrometric_attributes = AstrometricAttributes.from_objects(
            runs_by_age, runs_by_age[0], intersect_polygons)

        content_hash = hash_list(sorted(r.content_hash for r in contents.values()))

        camera_id_hash = _camera_id_hash_from(
            imaging_attributes, camera_consensus
        )
        attributes: dict[str, Any] = (
            asdict(chronology_attributes)
            | asdict(imaging_attributes)
            | asdict(astrometric_attributes)
            | {
                "content_hash": content_hash,
                "contents": contents,
                "state": frozendict(),
                "camera_id_hash": camera_id_hash,
            }
        )

        return cls(**attributes)

    def __hash__(self):
        assert self.content_hash, f"{type(self).__name__} has empty content_hash"
        return hash_list_int([self.content_hash])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseStack):
            return NotImplemented
        return (self.content_hash == other.content_hash) and (
            self.stable_hash == other.stable_hash
        )

    def __str__(self) -> str:
        return (
            f"Stack(Stable hash {self.stable_hash}, "
            f"Content hash {self.content_hash}, "
            f"Target {self.name}, "
            f"RA {units.ra_to_hms(self.center[0])}, "
            f"Dec {units.dec_to_dms(self.center[1])}, "
            f'Scale {self.image_scale:.2f}"/px, '
            f"PA {self.rotation_angle:.2f} deg, "
            f"Camera {self.camera}, "
            f"Exposure {self.exposure} s, "
            f"Integration {self.integration} s, "
            f"Filter {self.filter}, "
            f"Temp {self.mean_temperature:.02f} degC)"
        )


class CalibrationStack(BaseStack):
    """Calibration stack (bias/dark/flat) — no astrometric attributes."""

    @classmethod
    def from_run(
        cls,
        run: BaseRun,
        camera_consensus: tuple[int, ...] = (),
    ) -> "CalibrationStack":
        camera_id_hash = _camera_id_hash_from(run, camera_consensus)
        return cls(
            content_hash=run.content_hash,
            image_type=run.image_type,
            camera=run.camera,
            filter=run.filter,
            mean_temperature=run.mean_temperature,
            exposure=run.exposure,
            integration=run.integration,
            width_pixels=run.width_pixels,
            height_pixels=run.height_pixels,
            stable_hash=run.stable_hash,
            oldest_date=run.oldest_date,
            newest_date=run.newest_date,
            state=frozendict(),
            contents=RunsDict({run.stable_hash: run}),
            camera_id_hash=camera_id_hash,
        )

    def __hash__(self):
        assert self.content_hash, f"{type(self).__name__} has empty content_hash"
        return hash_list_int([self.content_hash])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseStack):
            return NotImplemented
        return (self.content_hash == other.content_hash) and (
            self.stable_hash == other.stable_hash
        )

    def __str__(self) -> str:
        return (
            f"CalibrationStack(Content hash {self.content_hash}, "
            f"Type {self.image_type}, "
            f"Camera {self.camera}, "
            f"Exposure {self.exposure}, "
            f"Integration {self.integration} s, "
            f"Filter {self.filter}, "
            f"Temp {self.mean_temperature:.02f} degC)"
        )


class StacksDict(dict[StackHash, ContextRef["Stack"]]):
    def __hash__(self):
        from cache.context_ref import dict_content_hash
        return hash_list_int([dict_content_hash(self)])

    def __str__(self):
        return f"StacksDict({len(self)})"


class CalibrationStacksDict(dict[StackHash, CalibrationStack]):
    def __hash__(self):
        return hash_list_int(
            [self[k].content_hash for k in sorted(self.keys())]
        )

    def __str__(self):
        return f"CalibrationStacksDict({len(self)})"
