"""Plain-string output-canvas labels.

The legacy `QualityLevel` enum and `Implementation` enum are gone:
pipelines compose ops directly (each op's `__name__` encodes its
backend), and per-pipeline output variants are encoded by the
identity's `filename_stem()` (e.g. `PlacedImageId.filename_stem()`
includes viewbox dimensions).

`compute_viewbox` accepts these as inputs so a pipeline body can
request a specific output-canvas size by name (`mosaic.viewbox(
"Master")` etc.). DOWNSCALE_FACTORS provides the canvas downscale
factor for each built-in name; user pipelines could pass any
string, but their canvas factor would default to 1.
"""
from typing import NewType

PreviewHash = NewType("PreviewHash", str)


# Output-canvas size labels.
SUBFRAME = "Subframe"
DOWNSCALED_MASTER = "DownscaledMaster"
MASTER = "Master"


# Per-canvas downscale factor used by `compute_viewbox`.
DOWNSCALE_FACTORS: dict[str, int] = {
    SUBFRAME: 1,
    DOWNSCALED_MASTER: 8,
    MASTER: 1,
}
