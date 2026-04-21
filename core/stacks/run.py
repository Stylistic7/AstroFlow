from dataclasses import dataclass, asdict
from typing import Any
from beartype import beartype
from frozendict import frozendict

from stacks.types import StackHash, DependenciesDict
from attributes import (
    AstroFlowStateAttributes,
    AstrometricAttributes,
    ChronologyAttributes,
    ImagingAttributes,
)
from metadata.classes import MetadataDict, FrameHash
from metadata.image_types import is_master
from hashing import hash_list_int
from cache.content_hashed import ContentHashed
from decorators import node
from cache.context_ref import ContextRef


@beartype
@dataclass(frozen=True)
class BaseRun(
    ContentHashed,
    ChronologyAttributes[StackHash],
    ImagingAttributes,
    AstroFlowStateAttributes,
):
    """Common base for all runs. Contains frames, metadata, and dependencies."""
    frames: tuple[FrameHash, ...]
    metadata: MetadataDict
    dependencies: DependenciesDict

    def __hash__(self) -> int:
        assert self.content_hash, f"{type(self).__name__} has empty content_hash"
        return hash_list_int([self.content_hash])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseRun):
            return NotImplemented
        return self.content_hash == other.content_hash


CalibrationRun = BaseRun


@beartype
@dataclass(frozen=True)
class Run(BaseRun, AstrometricAttributes):
    """A run of light frames — includes astrometry from plate solutions."""
    pass


class RunsDict(dict[str, BaseRun]):
    def __hash__(self):
        return hash_list_int(
            [self[k].content_hash for k in sorted(self.keys())]
        )

    def __str__(self):
        return f"RunsDict({len(self)})"


@node
def run_from_frames(
    frames: list[FrameHash], metadata: MetadataDict
) -> Run:
    chronology_attributes = ChronologyAttributes.from_frames(frames, metadata)
    imaging_attributes = ImagingAttributes.from_frames(frames, metadata)
    astrometric_attributes = AstrometricAttributes.from_frames(frames, metadata)

    frame_metadata = MetadataDict({h: i for h, i in metadata.items() if h in frames})
    assert frame_metadata
    assert not any(is_master(i.header) for i in frame_metadata.values())

    from hashing import hash_list
    content_hash = hash_list(sorted(frames))

    attributes: dict[str, Any] = (
        asdict(chronology_attributes)
        | asdict(imaging_attributes)
        | asdict(astrometric_attributes)
        | {
            "content_hash": content_hash,
            "frames": tuple(frames),
            "metadata": frame_metadata,
            "dependencies": frozendict(),
            "state": frozendict(),
        }
    )

    return Run(**attributes)


@node
def calibration_run_from_frames(
    frames: list[FrameHash], metadata: MetadataDict
) -> CalibrationRun:
    chronology_attributes = ChronologyAttributes.from_frames(frames, metadata)
    imaging_attributes = ImagingAttributes.from_frames(frames, metadata)

    frame_metadata = MetadataDict({h: i for h, i in metadata.items() if h in frames})
    assert frame_metadata
    assert not any(is_master(i.header) for i in frame_metadata.values())

    from hashing import hash_list
    content_hash = hash_list(sorted(frames))

    attributes: dict[str, Any] = (
        asdict(chronology_attributes)
        | asdict(imaging_attributes)
        | {
            "content_hash": content_hash,
            "frames": tuple(frames),
            "metadata": frame_metadata,
            "dependencies": frozendict(),
            "state": frozendict(),
        }
    )

    return CalibrationRun(**attributes)
