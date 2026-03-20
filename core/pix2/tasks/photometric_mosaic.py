"""PhotometricMosaic — join two images with photometric matching.

This is a script-based task invoked via -p parameter flags.
"""
from typing import Any

from pix2.image import Image
from pix2.recipe import Step
from pix2.tasks.copy_fits_keywords import CopyFITSKeywords

_SCRIPT_FILENAME = (
    "/Applications/PixInsight/src/scripts/"
    "JohnMurphy/PhotometricMosaic/AstroFlowPhotometricMosaic.js"
)

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
    "useTargetGradientCorrection": True,
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
        script_filename=_SCRIPT_FILENAME,
        parameters={k: str(v) for k, v in parameters.items()},
    ))

    result = Image(
        view_id="Mosaic",
        recipe=recipe,
        filter=images[0].filter,
    )
    CopyFITSKeywords(images[0], result)
    return result
