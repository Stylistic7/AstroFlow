from dataclasses import dataclass


@dataclass(frozen=True)
class ContentHashed:
  content_hash: str = ''
