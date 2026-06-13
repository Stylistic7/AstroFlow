"""HDRComposition — combine images of varying exposures into one HDR master.

PixInsight's HDRComposition replaces saturated regions of the longer
exposure with the shorter exposure scaled by the exposure ratio, then
feathers the seam. When no saturation is present it degrades to a
weighted blend, so it covers both intentional-HDR and accidental-mixed
exposure cases on the PixInsight path.

Inputs must be on the same pixel grid; pre-register them with
`StarAlignment` before composing.
"""
from typing import Any

from pix.image import Image
from pix.recipe import Step
from pix.tasks.js import Enum, execute_global


_DEFAULT_PARAMETERS = {
    "images": [],
    "inputHints": "",
    "outputHints": "",
    # FIXME(PJSR-2.0/V8): broken. On a fresh HDRComposition instance
    # `combinationMethod` reads undefined, and `HDRComposition
    # .HDRC_MaximumValue` is undefined too — the parameter and its enum
    # were renamed/removed in PixInsight 1.9.4. Needs per-process
    # migration to the new names (probe the live process). Not exercised
    # by the mono mosaic pipeline, so left flagged rather than guessed.
    "combinationMethod": Enum("HDRComposition.HDRC_MaximumValue"),
    "maskBinarizingThreshold": 0.80,
    "maskSmoothness": 8,
    "maskGrowth": 2,
    "replaceLargeScales": 0,
    "autoExposures": True,
    "rejectBlack": True,
    "useFittingRegion": False,
    "fittingRectX0": 0,
    "fittingRectY0": 0,
    "fittingRectX1": 0,
    "fittingRectY1": 0,
    "generate64BitResult": True,
    "outputMasks": False,
    "closePreviousImages": False,
}

_RENAME_JS = """
var windows = ImageWindow.windows;
windows[windows.length - 1].currentView.id = "{view_id}";
"""


def HDRComposition(
    images: list[Image],
    output_view_id: str = "HDR",
    **kwargs: Any,
) -> Image:
    """Compose multiple registered exposures into one HDR image.

    `images` should be sorted longest-exposure-first (HDRComposition
    treats the first image as the long-exposure base and replaces its
    saturated regions with subsequent shorter exposures).
    """
    if len(images) == 1:
        return images[0]

    recipe = images[0].recipe
    image_entries = [f'[true, "{img.view_id}"]' for img in images]
    parameters = _DEFAULT_PARAMETERS | kwargs | {
        "images": Enum(f"[{', '.join(image_entries)}]"),
    }

    js = execute_global("HDRComposition", parameters)
    js += "\n" + _RENAME_JS.format(view_id=output_view_id)

    recipe.add_step(Step(javascript=js, name="HDRComposition"))
    return Image(view_id=output_view_id, recipe=recipe)
