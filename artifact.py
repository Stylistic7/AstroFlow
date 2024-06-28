from dataclasses import dataclass


@dataclass(frozen=True)
class Artifact:
  key: str = None
  filename: str = None
  content_hash: str = None
