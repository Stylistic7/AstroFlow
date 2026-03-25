from dataclasses import dataclass, asdict
from typing import Any, Optional
from collections.abc import Sequence
import numpy as np
import ray
from beartype import beartype
from frozendict import frozendict

import units
from hashing import hash_list, hash_list_int
from stacks.simple_stack import Run, RunsDict, run_from_frames
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


@beartype
@dataclass(frozen=True)
class Stack(
    ContentHashed,
    ImagingAttributes,
    AstrometricAttributes,
    ChronologyAttributes[StackHash],
    AstroFlowStateAttributes,
):
    contents: RunsDict

    @classmethod
    def from_frames(
        cls, frames: Sequence[FrameHash], metadata: MetadataDict, max_run_gap
    ) -> Optional["Stack"]:
        metadata_list = [i for h, i in metadata.items() if h in frames]

        # Sort the light frames by observation date. The oldest image will be used
        # to compute the hash of the whole stack.
        sorted_metadata = sorted(metadata_list, key=lambda i: i.header["DATE-OBS"])

        contents = RunsDict()
        exposures = [
            (i.content_hash, i.header["EXPOSURE"])
            for i in metadata_list
            if "EXPOSURE" in i.header
        ]
        for exposure_cluster in cluster_continuous(exposures, epsilon=0.01):
            exposure_frames, exposures = zip(*exposure_cluster)
            assert np.all(np.isclose(exposures, exposures[0])), (
                f"exposures {exposures} are clustered, but not all close"
            )

            runs = cluster_continuous(
                [
                    (FrameHash(i.content_hash), i.header["DATE-OBS"])
                    for i in sorted_metadata
                    if i.content_hash in exposure_frames
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
        astrometric_attributes = AstrometricAttributes.from_frames(frames, metadata)

        attributes: dict[str, Any] = (
            asdict(chronology_attributes)
            | asdict(imaging_attributes)
            | asdict(astrometric_attributes)
            | {
                "contents": contents,
                "state": frozendict(),
            }
        )

        return cls(**attributes)

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

    def __hash__(self):
        return hash_list_int([self.content_hash])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Stack):
            return NotImplemented

        return (self.content_hash == other.content_hash) and (
            self.stable_hash == other.stable_hash
        )


class StacksDict(dict[StackHash, ContextRef["Stack"]]):
    def __hash__(self):
        from cache.context_ref import dict_content_hash
        return hash_list_int([dict_content_hash(self)])

    def __str__(self):
        return f"StacksDict({len(self)})"

