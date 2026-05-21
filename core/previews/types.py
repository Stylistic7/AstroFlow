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
    """Output-fidelity names. Kept as a StrEnum so existing
    `isinstance(v, QualityLevel)` checks in profiling/scheduling code
    continue to work; new code should use the bare string constants
    below directly. Pipelines never need to import this — they
    reference the modules-level strings, and the producer registry
    treats the level axis as just a string in its key tuples."""
    SUBFRAME = "Subframe"
    DOWNSCALED_MASTER = "DownscaledMaster"
    MASTER = "Master"
    POST_PROCESSED = "PostProcessed"
    # Gradient-corrected stack at downscaled resolution. Its own
    # artifact so a DownscaledMaster mosaic gets gradient removal
    # (cheaply, on the small image) independent of the full-res
    # PostProcessed master.
    DOWNSCALED_POST_PROCESSED = "DownscaledPostProcessed"


# Module-level aliases — these are the canonical references. New code
# imports `MASTER`/`SUBFRAME`/etc. directly without going through
# `QualityLevel.X`. Backed by the StrEnum members so equality checks
# still work.
SUBFRAME = QualityLevel.SUBFRAME
DOWNSCALED_MASTER = QualityLevel.DOWNSCALED_MASTER
MASTER = QualityLevel.MASTER
POST_PROCESSED = QualityLevel.POST_PROCESSED
DOWNSCALED_POST_PROCESSED = QualityLevel.DOWNSCALED_POST_PROCESSED


# Quality levels in priority order (best first → worst last). Web server
# endpoints walk this order when falling back to a lower quality.
QUALITY_PRIORITY = [POST_PROCESSED, MASTER, DOWNSCALED_MASTER, SUBFRAME]

# Set of all known quality-level values — used by profiling code to
# verify that a candidate string arg is actually a quality, not just
# any string. New profiling/scheduling code should prefer this over
# `isinstance(v, QualityLevel)` (it doesn't fire on arbitrary strings
# the way an `isinstance(v, str)` would).
ALL_QUALITY_LEVELS = [
    SUBFRAME, DOWNSCALED_MASTER, MASTER, POST_PROCESSED, DOWNSCALED_POST_PROCESSED,
]


# Integer downscale factor associated with each quality level.
DOWNSCALE_FACTORS: dict[str, int] = {
    SUBFRAME: 1,
    DOWNSCALED_MASTER: 8,
    MASTER: 1,
    POST_PROCESSED: 1,
    DOWNSCALED_POST_PROCESSED: 8,
}
