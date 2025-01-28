from dataclasses import dataclass, asdict, fields
from typing import Any

from context import context
from stacks.types import StackHash
from stacks.imaging_attributes import ImagingAttributes
from stacks.state_attributes import AstroFlowStateAttributes
from metadata.classes import MetadataDict, FrameHash
from hashing import hash_list, hash_list_int

DependenciesDict = dict[str, StackHash]


@dataclass(frozen=True)
class SimpleStack(ImagingAttributes, AstroFlowStateAttributes):
  content_hash: str
  frames: list[FrameHash]
  metadata: MetadataDict
  dependencies: DependenciesDict

  @classmethod
  def from_frames(cls,
                  frames: list[FrameHash],
                  metadata: MetadataDict | None = None) -> 'SimpleStack':
    metadata = metadata or context.metadata
    imaging_attributes = ImagingAttributes.from_frames(frames, metadata)

    frame_metadata = {h: i for h, i in metadata.items() if h in frames}
    assert frame_metadata

    attributes: dict[str, Any] = asdict(imaging_attributes) | {
        'content_hash': hash_list(frames),
        'frames': frames,
        'metadata': frame_metadata,
        'dependencies': DependenciesDict(),
        'state': {}
    }

    return cls(**attributes)

  def __str__(self) -> str:
    return (f'SimpleStack(Content hash {self.content_hash}, '
            f'Type {self.image_type}, '
            f'Camera {self.camera}, '
            f'Exposure {self.exposure_s}, '
            f'Integration {self.integration_s} s, '
            f'Filter {self.filter}, '
            f'Temp {self.mean_temperature:.02f} degC)')

  def __hash__(self):
    return hash_list_int([self.content_hash])

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, SimpleStack):
      return NotImplemented

    return (self.content_hash == other.content_hash)


class SimpleStacksDict(dict[str, SimpleStack]):

  def __hash__(self):
    return hash_list_int([self[k].content_hash for k in sorted(self.keys())])

  def __str__(self):
    return f'SimpleStacksDict({len(self)})'
