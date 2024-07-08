from dataclasses import dataclass
from typing import Optional

from plate_solving import PlateSolution


# TODO: Maybe unify this with Artifact, since they contain the same info.
@dataclass(frozen=True)
class ImageMetadata:
  content_hash: str = None
  filename: str = None
  header: dict = None
  plate_solution: Optional[PlateSolution] = None
