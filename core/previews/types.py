from typing import NewType
from enum import StrEnum

PreviewHash = NewType('PreviewHash', str)


class QualityLevel(StrEnum):
  SUBFRAME = "subframe"
  MASTER = "master"
  POST_PROCESSED = "post_processed"
