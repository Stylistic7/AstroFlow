from typing import NewType
from enum import StrEnum

PreviewHash = NewType("PreviewHash", str)


class QualityLevel(StrEnum):
    SUBFRAME = "Subframe"
    DOWNSCALED_MASTER = "DownscaledMaster"
    MASTER = "Master"
    POST_PROCESSED = "PostProcessed"
