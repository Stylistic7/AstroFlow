from typing import Any, cast

from pix.classes import ShellScriptTask, ImageList
from pix.context import append_task
from pix.processes import enum, execute_on_view_ids

_default_parameters = {
    'reference': 0.50,
    'lowThreshold': 0.20,
    'lowTolerance': 0.50,
    'highThreshold': 0.05,
    'highTolerance': 0.00,
    'iterations': 15,
    'scale': 5.00,
    'smoothness': 0.40,
    'downsamplingFactor': 16,
    'protection': True,
    'protectionThreshold': 0.10,
    'protectionAmount': 0.50,
    'protectionSmoothingFactor': 16,
    'lowClippingLevel': 0.000076,
    'automaticConvergence': False,
    'convergenceLimit': 0.00001000,
    'maxIterations': 10,
    'useSimplification': False,
    'simplificationDegree': 1,
    'simplificationScale': 1024,
    'generateSimpleModel': False,
    'generateGradientModel': False,
    'generateProtectionMasks': False,
    'gridSamplingDelta': 16,
}


def GradientCorrection(images: ImageList, **kwargs: Any) -> ImageList:
  assert all(i.view_id is not None for i in images)
  view_ids = [cast(str, i.view_id) for i in images]

  append_task(
      ShellScriptTask(javascript=execute_on_view_ids(
          'GradientCorrection', _default_parameters | kwargs, view_ids)))

  return images
