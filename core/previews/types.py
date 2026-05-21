from typing import NewType
from enum import StrEnum

PreviewHash = NewType("PreviewHash", str)


# Backend names — strings, not an enum. Used as the "implementation"
# axis in producer-registry keys and as path segments in on-disk file
# layout. Free-form by design: user pipelines may declare their own
# backend names without touching this module.
PIXINSIGHT = "PixInsight"
ASTROFLOW = "AstroFlow"
OSS = "OSS"
ALL_IMPLEMENTATIONS = [PIXINSIGHT, ASTROFLOW, OSS]


class QualityLevel(StrEnum):
    SUBFRAME = "Subframe"
    DOWNSCALED_MASTER = "DownscaledMaster"
    MASTER = "Master"
    POST_PROCESSED = "PostProcessed"
    # Gradient-corrected stack at downscaled resolution. Its own
    # artifact so a DownscaledMaster mosaic gets gradient removal
    # (cheaply, on the small image) independent of the full-res
    # PostProcessed master.
    DOWNSCALED_POST_PROCESSED = "DownscaledPostProcessed"


# Module-level aliases for ergonomic imports.
SUBFRAME = QualityLevel.SUBFRAME
DOWNSCALED_MASTER = QualityLevel.DOWNSCALED_MASTER
MASTER = QualityLevel.MASTER
POST_PROCESSED = QualityLevel.POST_PROCESSED
DOWNSCALED_POST_PROCESSED = QualityLevel.DOWNSCALED_POST_PROCESSED


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
    QualityLevel.DOWNSCALED_POST_PROCESSED: 8,
}
