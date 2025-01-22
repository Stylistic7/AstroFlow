from typing import Any
import os

from pix.classes import ShellScriptTask, Image, ImageList
from pix.context import append_task, add_images_to_save

# TODO: These parameters must be floats, not strings, so they can be overriden
# by the user.
_default_parameters = {
    'referenceView': 'placed_0',
    'targetView': 'placed_1',
    'replaceRefImage': False,
    'pixelSize': 3.75999999046326,
    'focalLength': 540,
    'refLogStarDetection': -1,
    'tgtLogStarDetection': -1,
    'starFluxTolerance': 1.5,
    'starSearchRadius': 2.5,
    'apertureGrowthRate': 0.25,
    'apertureAdd': 1,
    'apertureGap': 1,
    'apertureBgDelta': 19,
    'limitPhotoStarsPercent': 100,
    'linearRangeRef': 0.7,
    'linearRangeTgt': 0.7,
    'outlierRemovalPercent': 2,
    'useAutoPhotometry': True,
    'joinSize': 90,
    'joinPosition': 0,
    'joinOutlierPercent': 2,
    'cropTargetPreviewLeft': 0,
    'cropTargetPreviewTop': 0,
    'cropTargetPreviewWidth': 1,
    'cropTargetPreviewHeight': 1,
    'useCropTargetToReplaceRegion': False,
    'sampleStarGrowthRate': 0.25,
    'sampleStarGrowthRateTarget': 1,
    'limitSampleStarsPercent': 100,
    'sampleSize': 27,
    'maxSamples': 3000,
    'useAutoSampleGeneration': True,
    'adjustScale0': 1,
    'adjustScale1': 1,
    'adjustScale2': 1,
    'adjustScaleLineOffset': 0,
    'overlapGradientSmoothness': -1,
    'taperLength': '47',
    'useAutoTaperLength': True,
    'useTargetGradientCorrection': True,
    'targetGradientSmoothness': 2,
    'limitMaskStarsPercent': 10,
    'maskStarGrowthRate': 0.25,
    'maskStarGrowthLimit': 400,
    'maskStarRadiusAdd': 4,
    'useAutoMaskStarSize': True,
    'useMosaicOverlay': False,
    'useMosaicRandom': True,
    'useMosaicAverage': False,
    'createJoinMask': True,
    'smallScreen': False,
    'graphWidth': 1200,
    'graphHeight': 800,
    'extraControls': False
}


def PhotometricMosaic(images: ImageList, output_filename: str,
                      **kwargs: Any) -> ImageList:
  print(f'### Entering PhotometricMosaic, images {images} '
        f'output_filename {output_filename}')
  parameters = _default_parameters | kwargs
  javascript_filename = "/Applications/PixInsight/src/scripts/JohnMurphy/PhotometricMosaic/AstroFlowPhotometricMosaic.js"

  parameters['referenceView'] = images[0].view_id
  parameters['targetView'] = images[1].view_id

  # TODO: The `javascript_filename` thing is long. Maybe there can be two
  # ShellScriptTask types, one that accepts code, and one that accepts a filename.
  append_task(
      ShellScriptTask(javascript_filename=javascript_filename,
                      parameters=parameters))

  images = [
      Image(filename=output_filename,
            view_id='Mosaic',
            filter=images[0].filter,
            dirty=True)
  ]

  add_images_to_save(images)

  print(f'### PhotometricMosaic returning {images}')
  return images
