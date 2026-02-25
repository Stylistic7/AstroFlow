from dataclasses import dataclass
import datetime
from collections.abc import Sequence
from typing import Generic, TypeVar
from beartype import beartype

from metadata.classes import MetadataDict, FrameHash

_HashType = TypeVar("_HashType", bound=str)


@beartype
@dataclass(frozen=True)
class ChronologyAttributes(Generic[_HashType]):
    stable_hash: _HashType
    oldest_date: datetime.datetime
    newest_date: datetime.datetime

    @classmethod
    def from_frames(
        cls, frames: Sequence[FrameHash], metadata: MetadataDict
    ) -> "ChronologyAttributes":
        metadata_list = [
            i for i in metadata.values() if i.content_hash in frames
        ]

        sorted_metadata = sorted(
            metadata_list, key=lambda i: i.header["DATE-OBS"]
        )
        oldest_date = sorted_metadata[0].header["DATE-OBS"]
        newest_date = sorted_metadata[-1].header["DATE-OBS"]

        # The stable hash is just the content hash of the oldest frame.
        stable_hash = sorted_metadata[0].content_hash

        return cls(stable_hash, oldest_date, newest_date)

    @classmethod
    def from_objects(
        cls, objects: Sequence["ChronologyAttributes"]
    ) -> "ChronologyAttributes":
        oldest_date = min(o.oldest_date for o in objects)
        newest_date = max(o.newest_date for o in objects)

        sorted_objects = sorted(objects, key=lambda i: i.oldest_date)
        stable_hash = sorted_objects[0].stable_hash

        return cls(stable_hash, oldest_date, newest_date)

    def __lt__(self, other):
        if not isinstance(other, ChronologyAttributes):
            return NotImplemented
        return self.oldest_date < other.oldest_date
