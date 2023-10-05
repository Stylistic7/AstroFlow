import os

from .decorators import accepts_view_ids
from .classes import ShellScriptTask, ViewIdList
from .context import append_task

# TODO: These parameters must be floats, not strings, so they can be overriden
# by the user.
_default_parameters = {
    'referenceView': 'mosaic_0',
    'targetView': 'mosaic_1',
    'replaceRefImage': 'false',
    'pixelSize': '3.75999999046326',
    'focalLength': '540',
    'refLogStarDetection': '-1',
    'tgtLogStarDetection': '-1',
    'starFluxTolerance': '1.5',
    'starSearchRadius': '2.5',
    'apertureGrowthRate': '0.25',
    'apertureAdd': '1',
    'apertureGap': '1',
    'apertureBgDelta': '19',
    'limitPhotoStarsPercent': '100',
    'linearRangeRef': '0.681',
    'linearRangeTgt': '0.683',
    'outlierRemovalPercent': '2',
    'useAutoPhotometry': 'true',
    'joinSize': '90',
    'joinPosition': '0',
    'joinOutlierPercent': '2',
    'cropTargetPreviewLeft': '0',
    'cropTargetPreviewTop': '0',
    'cropTargetPreviewWidth': '1',
    'cropTargetPreviewHeight': '1',
    'useCropTargetToReplaceRegion': 'false',
    'sampleStarGrowthRate': '0.25',
    'sampleStarGrowthRateTarget': '1',
    'limitSampleStarsPercent': '15.329999923706055',
    'sampleSize': '27',
    'maxSamples': '3000',
    'useAutoSampleGeneration': 'true',
    'adjustScale0': '1',
    'adjustScale1': '1',
    'adjustScale2': '1',
    'adjustScaleLineOffset': '0',
    'overlapGradientSmoothness': '-1',
    'taperLength': '47',
    'useAutoTaperLength': 'true',
    'useTargetGradientCorrection': 'true',
    'targetGradientSmoothness': '2',
    'limitMaskStarsPercent': '10',
    'maskStarGrowthRate': '0.25',
    'maskStarGrowthLimit': '400',
    'maskStarRadiusAdd': '4',
    'useAutoMaskStarSize': 'true',
    'useMosaicOverlay': 'false',
    'useMosaicRandom': 'true',
    'useMosaicAverage': 'false',
    'createJoinMask': 'true',
    'smallScreen': 'false',
    'graphWidth': '1200',
    'graphHeight': '800',
    'extraControls': 'false'
}


@accepts_view_ids
def PhotometricMosaic(view_ids, **kwargs):
  parameters = _default_parameters | kwargs
  javascript_filename = "/Applications/PixInsight/src/scripts/JohnMurphy/mosaic/src/AstroFlowPhotometricMosaic.js"

  parameters['referenceView'] = view_ids[0]
  parameters['targetView'] = view_ids[1]

  # TODO: The `javascript_filename` thing is long. Maybe there can be two
  # ShellScriptTask types, one that accepts code, and one that accepts a filename.
  append_task(
      ShellScriptTask(javascript_filename=javascript_filename,
                      parameters=parameters))

  return ViewIdList(['Mosaic'])
