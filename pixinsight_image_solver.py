import os

import pixinsight_shell_script

_default_parameters = {
    'metadata_focal': '541.5685232722849',
    'metadata_useFocal': 'true',
    'metadata_xpixsz': '3.759999990463257',
    'metadata_resolution': '0.0003977929324021598',
    'metadata_referenceSystem': 'ICRS',
    'metadata_ra': '311.71182622264297',
    'metadata_dec': '30.854210446991924',
    'metadata_epoch': '2460126.7458429397',
    'metadata_observationTime': '2460127.3486342593',
    'metadata_topocentric': 'false',
    'solver_version': '5.6.6',
    'solver_magnitude': '12',
    'solver_autoMagnitude': 'true',
    'solver_generateErrorImg': 'false',
    'solver_structureLayers': '5',
    'solver_minStructureSize': '0',
    'solver_hotPixelFilterRadius': '1',
    'solver_noiseReductionFilterRadius': '0',
    'solver_sensitivity': '0.5',
    'solver_peakResponse': '0.5',
    'solver_brightThreshold': '3',
    'solver_maxStarDistortion': '0.6',
    'solver_psfSearchRadius': '12',
    'solver_catalogMode': '3',
    'solver_vizierServer': 'http://vizier.u-strasbg.fr/',
    'solver_showStars': 'false',
    'solver_showSimplifiedSurfaces': 'false',
    'solver_showDistortion': 'false',
    'solver_generateDistortModel': 'false',
    'solver_catalog': 'GaiaDR3_XPSD',
    'solver_distortionCorrection': 'true',
    'solver_splineSmoothing': '0.014999999664723873',
    'solver_enableSimplifier': 'true',
    'solver_simplifierTolerance': '0.05000000074505806',
    'solver_simplifierRejectFraction': '0.10000000149011612',
    'solver_useDistortionModel': 'false',
    'solver_onlyOptimize': 'false',
    'solver_alignAlgorithm': '0',
    'solver_useActive': 'false',
    'solver_outSuffix': '_WCS',
    'solver_files': '',
    'solver_projection': '0',
    'solver_projectionOriginMode': '0',
    'solver_distortedCorners': 'false',
    'solver_optimizeSolution': 'true',
    'solver_restrictToHQStars': 'false',
    'non_interactive': 'true'
}


def run(filenames, **kwargs):
  parameters = _default_parameters | kwargs
  javascript_filename = "/Applications/PixInsight/src/scripts/AdP/ImageSolver.js"

  # Remove output filenames that already exist.
  output_filenames = [f'{os.path.splitext(f)[0]}_WCS.xisf' for f in filenames]
  filenames = [
      i for i, o in zip(filenames, output_filenames) if not os.path.exists(o)
  ]

  parameters['solver_files'] = '|'.join(filenames)

  task = pixinsight_shell_script.PixInsightShellScriptTask(
      javascript_filename, parameters)

  pixinsight_shell_script.run_script([task])

  assert (all(os.path.exists(f) for f in output_filenames))
  return output_filenames


if __name__ == '__main__':
  filenames = [
      '/Users/nobody/Downloads/Veil Pano/master_light_Filamentary_nebula_RA_311_5_DEC_30_2_C11_H_v1_post_registered.xisf',
      '/Users/nobody/Downloads/Veil Pano/master_light_Filamentary_nebula_RA_313_1_DEC_30_0_C11_H_v1_post_registered.xisf'
  ]

  run(filenames)