"""PhotometricMosaic — join two images with photometric matching.

This is a script-based task invoked via -p parameter flags.
"""
import re
from typing import Any

from pix.image import Image
from pix.recipe import Step
from pix.tasks.copy_fits_keywords import CopyFITSKeywords
from third_party_tools import pixinsight

def _script_filename() -> str:
    return pixinsight("src", "scripts", "JohnMurphy", "PhotometricMosaic", "AstroFlowPhotometricMosaic.js")

_DEFAULT_PARAMETERS = {
    "referenceView": "placed_0",
    "targetView": "placed_1",
    "replaceRefImage": False,
    "pixelSize": 3.75999999046326,
    "focalLength": 540,
    "refLogStarDetection": -1,
    "tgtLogStarDetection": -1,
    "starFluxTolerance": 1.5,
    "starSearchRadius": 2.5,
    "apertureGrowthRate": 0.25,
    "apertureAdd": 1,
    "apertureGap": 1,
    "apertureBgDelta": 19,
    "limitPhotoStarsPercent": 100,
    "linearRangeRef": 0.7,
    "linearRangeTgt": 0.7,
    "outlierRemovalPercent": 2,
    "useAutoPhotometry": True,
    "joinSize": 90,
    "joinPosition": 0,
    "joinOutlierPercent": 2,
    "useCropTargetToReplaceRegion": False,
    "sampleStarGrowthRate": 0.25,
    "sampleStarGrowthRateTarget": 1,
    "limitSampleStarsPercent": 23,
    "sampleSize": 27,
    "maxSamples": 3000,
    "useAutoSampleGeneration": True,
    "overlapGradientSmoothness": -1,
    "taperLength": "47",
    "useAutoTaperLength": True,
    # AstroFlow runs MultiscaleGradientCorrection per panel BEFORE PMM, so
    # the panels handed to PMM are already gradient-free. PMM's own
    # target-gradient correction then has no real gradient to fit and
    # instead extrapolates its surface-spline across the panel, producing
    # a background glow in the corner farthest from the overlap. Disable
    # it; PMM still applies the photometric scale + the overlap-join
    # gradient, so the seam stays matched.
    "useTargetGradientCorrection": False,
    "targetGradientSmoothness": 2,
    "limitMaskStarsPercent": 10,
    "maskStarGrowthRate": 0.25,
    "maskStarGrowthLimit": 400,
    "maskStarRadiusAdd": 4,
    "useAutoMaskStarSize": True,
    "useMosaicOverlay": True,
    "useMosaicRandom": False,
    "useMosaicAverage": False,
    "createJoinMask": True,
    "smallScreen": False,
    "graphWidth": 1200,
    "graphHeight": 800,
    "extraControls": False,
}


def PhotometricMosaic(images: list[Image], **kwargs: Any) -> Image:
    """Join two images using PixInsight's PhotometricMosaic script."""
    assert len(images) == 2
    recipe = images[0].recipe

    parameters = _DEFAULT_PARAMETERS | kwargs | {
        "referenceView": images[0].view_id,
        "targetView": images[1].view_id,
    }

    recipe.add_step(Step(
        name="PhotometricMosaic",
        script_filename=_script_filename(),
        parameters={k: str(v) for k, v in parameters.items()},
        # astroflow_main() logs any exception (console.criticalln) and returns
        # normally, so a failed join never creates the Mosaic view. Assert
        # the script's conclusive success line ("Created Mosaic …", emitted
        # only after the view is built) so a silent failure raises here
        # instead of surfacing later as a confusing missing-output error at
        # save. The real cause now appears in the recipe log (was previously
        # swallowed by a GUI MessageBox on the undefined StdIcon_Error).
        success_pattern=re.compile(r"Created\s+Mosaic\b"),
    ))

    result = Image(
        view_id="Mosaic",
        recipe=recipe,
        filter=images[0].filter,
    )
    CopyFITSKeywords(images[0], result)
    return result
