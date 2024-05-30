from .classes import ShellScriptTask
from .context import append_task
from .processes import enum, execute_global

_default_parameters = {
    'structureLayers': 5,
    'noiseLayers': 0,
    'hotPixelFilterRadius': 1,
    'noiseReductionFilterRadius': 0,
    'minStructureSize': 0,
    'sensitivity': 0.50,
    'peakResponse': 0.50,
    'brightThreshold': 3.00,
    'maxStarDistortion': 0.60,
    'allowClusteredSources': False,
    'localMaximaDetectionLimit': 0.75,
    'upperLimit': 1.000,
    'invert': False,
    'distortionModel': '',
    'undistortedReference': False,
    'distortionCorrection': True,
    'distortionMaxIterations': 20,
    'distortionTolerance': 0.005,
    'distortionAmplitude': 2,
    'localDistortion': True,
    'localDistortionScale': 256,
    'localDistortionTolerance': 0.050,
    'localDistortionRejection': 2.50,
    'localDistortionRejectionWindow': 64,
    'localDistortionRegularization': 0.010,
    'matcherTolerance': 0.0500,
    'ransacTolerance': 2.00,
    'ransacMaxIterations': 2000,
    'ransacMaximizeInliers': 1.00,
    'ransacMaximizeOverlapping': 1.00,
    'ransacMaximizeRegularity': 1.00,
    'ransacMinimizeError': 1.00,
    'maxStars': 0,
    'fitPSF': enum('StarAlignment.prototype.FitPSF_DistortionOnly'),
    'psfTolerance': 0.50,
    'useTriangles': False,
    'polygonSides': 5,
    'descriptorsPerStar': 20,
    'restrictToPreviews': True,
    'intersection': enum('StarAlignment.prototype.MosaicOnly'),
    'useBrightnessRelations': False,
    'useScaleDifferences': False,
    'scaleTolerance': 0.100,
    'referenceImage': '',
    'referenceIsFile': False,
    'targets': [],
    'inputHints': '',
    'outputHints': '',
    'mode': enum('StarAlignment.prototype.RegisterMatch'),
    'writeKeywords': True,
    'generateMasks': False,
    'generateDrizzleData': True,
    'generateDistortionMaps': False,
    'inheritAstrometricSolution': True,
    'frameAdaptation': False,
    'randomizeMosaic': False,
    'noGUIMessages': True,
    'useSurfaceSplines': True,
    'extrapolateLocalDistortion': True,
    'splineSmoothness': 0.050,
    'pixelInterpolation': enum('StarAlignment.prototype.Auto'),
    'clampingThreshold': 0.30,
    'outputDirectory': '',
    'outputExtension': ".xisf",
    'outputPrefix': '',
    'outputPostfix': "_r",
    'maskPostfix': "_m",
    'distortionMapPostfix': "_dm",
    'outputSampleFormat': enum('StarAlignment.prototype.SameAsTarget'),
    'overwriteExistingFiles': False,
    'onError': enum('StarAlignment.prototype.Continue'),
    'useFileThreads': True,
    'fileThreadOverload': 1.00,
    'maxFileReadThreads': 0,
    'maxFileWriteThreads': 0,
    'memoryLoadControl': True,
    'memoryLoadLimit': 0.85,
    'outputData': [],
}

# outputData will contain:
# outputImage, outputMask, totalPairMatches, inliers, overlapping, regularity, quality, rmsError, rmsErrorDev, peakErrorX, peakErrorY, H11, H12, H13, H21, H22, H23, H31, H32, H33, frameAdaptationBiasRK, frameAdaptationBiasG, frameAdaptationBiasB, frameAdaptationSlopeRK, frameAdaptationSlopeG, frameAdaptationSlopeB, frameAdaptationAvgDevRK, frameAdaptationAvgDevG, frameAdaptationAvgDevB, referenceStarX, referenceStarY, targetStarX, targetStarY, totalLocalDistortionMatches, referenceLocalDistortionX, referenceLocalDistortionY, targetLocalDistortionX, targetLocalDistortionY, targetLocalDistortionDX, targetLocalDistortionDY, localDistortionWeights, outputDistortionMap, localDistortionMinDeltaX, localDistortionMinDeltaY, localDistortionMaxDeltaX, localDistortionMaxDeltaY

_rename_reference_view = """
View.viewById("{old_view_id}").id = "{new_view_id}";
"""

_close_views = """
var view_ids = {view_ids};

for (let i = 0; i < view_ids.length; ++i) {{
  let window = View.viewById(view_ids[i]).window;
  window.forceClose();
}}
"""


# TODO: The reference image is the first view_id, but this is not obvious.
def StarAlignment(view_ids, close_originals=True, **kwargs):
  parameters = _default_parameters | kwargs
  parameters['referenceImage'] = view_ids[0]

  # TODO: This is horrifically ugly.
  targets = [f'[true, false, "{v}"]' for v in view_ids[1:]]
  parameters['targets'] = enum(f'[{", ".join(targets)}]')

  append_task(
      ShellScriptTask(javascript=execute_global('StarAlignment', parameters)))

  rename_reference_view = _rename_reference_view.format(
      old_view_id=view_ids[0], new_view_id=view_ids[0] + '_registered')
  append_task(ShellScriptTask(javascript=rename_reference_view))

  if close_originals:
    close_views = _close_views.format(view_ids=view_ids[1:])
    append_task(ShellScriptTask(javascript=close_views))

  output_view_ids = [v + '_registered' for v in view_ids]
  return output_view_ids
