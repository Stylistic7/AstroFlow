"""Plain-string output-canvas labels.

The legacy `QualityLevel` enum and `Implementation` enum are gone:
pipelines compose ops directly (each op's `__name__` encodes its
backend), and per-pipeline output variants are encoded by the
identity's `filename_stem()` (e.g. `PlacedImageId.filename_stem()`
includes viewbox dimensions).

`compute_viewbox` accepts these as inputs so a pipeline body can
request a specific output-canvas size by name (`mosaic.viewbox(
"Master")` etc.): a Subframe canvas is a fixed-size thumbnail, any
other is at the object's native image scale.
"""
from typing import NewType

PreviewHash = NewType("PreviewHash", str)


# Output-canvas size labels.
SUBFRAME = "Subframe"
MASTER = "Master"
