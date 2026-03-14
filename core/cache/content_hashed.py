from dataclasses import dataclass
from protocols import ContentHash
from beartype import beartype


@beartype
@dataclass(frozen=True, kw_only=True)
class ContentHashed:
    content_hash: ContentHash = ""
