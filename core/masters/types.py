from dataclasses import dataclass, asdict
from typing import Any

from stacks.types import StackHash
from attributes import AstroFlowStateAttributes, ChronologyAttributes, ImagingAttributes
from hashing import hash_list_int
from metadata.classes import ImageMetadata, MetadataDict, FrameHash
from cache.content_hashed import ContentHashed


# TODO: Should Master also include AstrometricAttributes?
@dataclass(frozen=True)
class Master(ContentHashed, ChronologyAttributes[StackHash], ImagingAttributes,
             AstroFlowStateAttributes):

  # TODO: Should Master inherit from ImageMetadata?
  metadata: ImageMetadata

  @classmethod
  def from_frame(cls, frame: FrameHash, metadata: MetadataDict) -> 'Master':
    chronology_attributes = ChronologyAttributes.from_frames([frame], metadata)
    imaging_attributes = ImagingAttributes.from_frames([frame], metadata)

    frame_metadata = metadata[frame]

    # TODO: Unify CONTHASH and STBLHASH somewhere.
    attributes: dict[
        str,
        Any] = asdict(chronology_attributes) | asdict(imaging_attributes) | {
            'content_hash': frame_metadata.header['CONTHASH'],
            'stable_hash': frame_metadata.header['STBLHASH'],
            'metadata': frame_metadata,
            'state': {}
        }

    return cls(**attributes)

  def __hash__(self) -> int:
    return hash_list_int([self.content_hash])


class MastersDict(dict[str, Master]):

  def __hash__(self):
    return hash_list_int([self[k].content_hash for k in sorted(self.keys())])

  def __str__(self):
    return f'MastersDict({len(self)})'
