from dataclasses import dataclass
from protocols import ContentHash


@dataclass(frozen=True, kw_only=True)
class ContentHashed:
    content_hash: ContentHash = ""
