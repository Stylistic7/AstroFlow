from dataclasses import dataclass, asdict
from typing import Any

from context import context
from stacks.types import DependenciesDict
from stacks.imaging_attributes import ImagingAttributes
from stacks.state_attributes import AstroFlowStateAttributes
from metadata.classes import MetadataDict, FrameHash
from hashing import hash_list


@dataclass(frozen=True)
class SimpleStack(ImagingAttributes, AstroFlowStateAttributes):
  content_hash: str
  frames: list[FrameHash]
  dependencies: DependenciesDict

  @classmethod
  def from_frames(cls,
                  frames: list[FrameHash],
                  metadata: MetadataDict | None = None) -> 'SimpleStack':
    metadata = metadata or context.metadata
    imaging_attributes = ImagingAttributes.from_frames(frames, metadata)

    attributes: dict[str, Any] = asdict(imaging_attributes) | {
        'content_hash': hash_list(frames),
        'frames': frames,
        'dependencies': DependenciesDict(),
        'state': {}
    }

    return cls(**attributes)

  def __str__(self) -> str:
    return (f'Stack(Content hash {self.content_hash}, '
            f'Type {self.image_type}, '
            f'Camera {self.camera}, '
            f'Exposure {self.exposure_s}, '
            f'Integration {self.integration_s} s, '
            f'Filter {self.filter}, '
            f'Temp {self.mean_temperature:.02f} degC)')

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, SimpleStack):
      return NotImplemented

    return (self.content_hash == other.content_hash)
