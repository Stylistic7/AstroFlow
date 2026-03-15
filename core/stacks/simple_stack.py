from dataclasses import dataclass, asdict
from typing import Any
from beartype import beartype

from stacks.types import StackHash, DependenciesDict
from attributes import (
    AstroFlowStateAttributes,
    ChronologyAttributes,
    ImagingAttributes,
)
from metadata.classes import MetadataDict, FrameHash
from metadata.image_types import is_master
from hashing import hash_list_int
from cache.content_hashed import ContentHashed
from decorators import cached
from hashing.types import LargeObject


@beartype
@dataclass(frozen=True)
class SimpleStack(
    ContentHashed,
    ChronologyAttributes[StackHash],
    ImagingAttributes,
    AstroFlowStateAttributes,
):
    frames: list[FrameHash]
    metadata: MetadataDict
    dependencies: DependenciesDict

    def __str__(self) -> str:
        return (
            f"SimpleStack(Content hash {self.content_hash}, "
            f"Type {self.image_type}, "
            f"Camera {self.camera}, "
            f"Exposure {self.exposure_s}, "
            f"Integration {self.integration_s} s, "
            f"Filter {self.filter}, "
            f"Temp {self.mean_temperature:.02f} degC)"
        )

    def __hash__(self) -> int:
        return hash_list_int([self.content_hash])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SimpleStack):
            return NotImplemented

        return self.content_hash == other.content_hash


class SimpleStacksDict(dict[str, SimpleStack]):
    def __hash__(self):
        return hash_list_int(
            [self[k].content_hash for k in sorted(self.keys())]
        )

    def __str__(self):
        return f"SimpleStacksDict({len(self)})"


class CalibrationStack(SimpleStack):
    pass


@cached.remote
def simple_stack_from_frames(
    frames: list[FrameHash], metadata: MetadataDict
) -> SimpleStack:
    chronology_attributes = ChronologyAttributes.from_frames(frames, metadata)
    imaging_attributes = ImagingAttributes.from_frames(frames, metadata)

    frame_metadata = MetadataDict({h: i for h, i in metadata.items() if h in frames})
    assert frame_metadata
    assert not any(is_master(i.header) for i in frame_metadata.values())

    attributes: dict[str, Any] = (
        asdict(chronology_attributes)
        | asdict(imaging_attributes)
        | {
            "frames": frames,
            "metadata": frame_metadata,
            "dependencies": DependenciesDict(),
            "state": {},
        }
    )

    return SimpleStack(**attributes)
