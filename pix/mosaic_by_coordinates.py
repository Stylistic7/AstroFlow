import os

from .decorators import accepts_filenames
from .classes import ShellScriptTask, FileList
from .context import append_task

_default_parameters = {
    'engine_files': '""',
    'engine_centerCoordsAuto': True,
    'engine_centerRA': 312.893418127648,
    'engine_centerDec': 30.863823588048277,
    'engine_resolutionAuto': True,
    'engine_resolution': 0.00019886620125921314,
    'engine_rotationAuto': True,
    'engine_rotation': 89.68771925250053,
    'engine_projectionAuto': False,
    'engine_projection': 3,
    'engine_projectionOriginMode': 0,
    'engine_projectionOriginRA': 0,
    'engine_projectionOriginDec': 0,
    'engine_dimensionsAuto': True,
    'engine_width': 19307,
    'engine_height': 23094,
    'engine_qualityHQ': True,
    'engine_pixelInterpolation': 5,
    'engine_clampingThreshold': 0.3,
    'engine_suffix': '"_registered"',
    'engine_overwrite': False,
    'engine_errorPolicy': 1,
    'engine_outputDir': '""',
    'non_interactive': True
}


@accepts_filenames
def MosaicByCoordinates(filenames, **kwargs):
  parameters = _default_parameters | kwargs
  javascript_filename = "/Applications/PixInsight/src/scripts/AdP/MosaicByCoordinates.js"

  output_filenames = [
      f'{os.path.splitext(f)[0]}_registered.xisf' for f in filenames
  ]

  # Remove output filenames that already exist.
  # filenames = [
  #     i for i, o in zip(filenames, output_filenames) if not os.path.exists(o)
  # ]

  parameters['engine_files'] = '|'.join(filenames)

  append_task(
      ShellScriptTask(javascript_filename=javascript_filename,
                      parameters=parameters))

  return FileList(output_filenames)
