from typing import NewType
from enum import StrEnum

PreviewHash = NewType("PreviewHash", str)


class Implementation(StrEnum):
    PIXINSIGHT = "PixInsight"
    ASTROFLOW = "AstroFlow"
    OSS = "OSS"


PIXINSIGHT = Implementation.PIXINSIGHT
ASTROFLOW = Implementation.ASTROFLOW
OSS = Implementation.OSS
ALL_IMPLEMENTATIONS = list(Implementation)


class QualityLevel(StrEnum):
    SUBFRAME = "Subframe"
    DOWNSCALED_MASTER = "DownscaledMaster"
    MASTER = "Master"
    POST_PROCESSED = "PostProcessed"


# Module-level aliases for ergonomic imports.
SUBFRAME = QualityLevel.SUBFRAME
DOWNSCALED_MASTER = QualityLevel.DOWNSCALED_MASTER
MASTER = QualityLevel.MASTER
POST_PROCESSED = QualityLevel.POST_PROCESSED


# Quality levels in priority order (best first → worst last). Web server
# endpoints walk this order when falling back to a lower quality.
QUALITY_PRIORITY = [POST_PROCESSED, MASTER, DOWNSCALED_MASTER, SUBFRAME]


# Integer downscale factor associated with each QualityLevel.
# DOWNSCALED_MASTER is always 8x; everything else is full resolution.
DOWNSCALE_FACTORS: dict[QualityLevel, int] = {
    QualityLevel.SUBFRAME: 1,
    QualityLevel.DOWNSCALED_MASTER: 8,
    QualityLevel.MASTER: 1,
    QualityLevel.POST_PROCESSED: 1,
}
