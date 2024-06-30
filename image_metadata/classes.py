from dataclasses import dataclass
from typing import Optional

from plate_solving import PlateSolution


@dataclass(frozen=True)
class ImageMetadata:
  content_hash: str = None
  filename: str = None
  header: dict = None
  plate_solution: Optional[PlateSolution] = None
