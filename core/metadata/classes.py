from dataclasses import dataclass
from typing import NewType

from hashing import hash_list_int
from plate_solving import PlateSolution

FrameHash = NewType("FrameHash", str)


# TODO: Maybe unify this with Artifact, since they contain the same info.
@dataclass(frozen=True)
class ImageMetadata:
    content_hash: str
    filename: str
    header: dict
    plate_solution: PlateSolution | None = None


class MetadataDict(dict[FrameHash, ImageMetadata]):
    def __hash__(self):
        return hash_list_int(sorted(self.keys()))

    def __str__(self):
        return f"MetadataDict({len(self)})"
