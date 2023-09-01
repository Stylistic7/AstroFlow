import pixinsight_shell_script

_default_parameters = {
    'engine_files': '',
    'engine_centerCoordsAuto': 'true',
    'engine_centerRA': '312.893418127648',
    'engine_centerDec': '30.863823588048277',
    'engine_resolutionAuto': 'true',
    'engine_resolution': '0.00019886620125921314',
    'engine_rotationAuto': 'true',
    'engine_rotation': '89.68771925250053',
    'engine_projectionAuto': 'false',
    'engine_projection': '3',
    'engine_projectionOriginMode': '0',
    'engine_projectionOriginRA': '0',
    'engine_projectionOriginDec': '0',
    'engine_dimensionsAuto': 'true',
    'engine_width': '19307',
    'engine_height': '23094',
    'engine_qualityHQ': 'true',
    'engine_pixelInterpolation': '5',
    'engine_clampingThreshold': '0.3',
    'engine_suffix': '_registered',
    'engine_overwrite': 'false',
    'engine_errorPolicy': '1',
    'engine_outputDir': '',
    'non_interactive': 'true'
}


def run(filenames, **kwargs):
  parameters = _default_parameters | kwargs
  javascript_filename = "/Applications/PixInsight/src/scripts/AdP/MosaicByCoordinates.js"

  # Remove output filenames that already exist.
  output_filenames = [
      f'{os.path.splitext(f)[0]}_registered.xisf' for f in filenames
  ]
  filenames = [
      i for i, o in zip(filenames, output_filenames) if not os.path.exists(o)
  ]

  parameters['engine_files'] = '|'.join(filenames)

  task = pixinsight_shell_script.PixInsightShellScriptTask(
      javascript_filename, parameters)

  pixinsight_shell_script.run_script([task])

  assert (all(os.path.exists(f) for f in output_filenames))
  return output_filenames
