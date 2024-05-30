from .classes import ShellScriptTask
from .context import append_task
from .processes import enum, execute_on_view_ids

_default_parameters = {
    'zoomFactor': -2,
    'downsamplingMode': enum('IntegerResample.prototype.Average'),
    'xResolution': 72.000,
    'yResolution': 72.000,
    'metric': False,
    'forceResolution': False,
    'gammaCorrection': False,
    'noGUIMessages': True,
}


def IntegerResample(view_ids, **kwargs):
  append_task(
      ShellScriptTask(javascript=execute_on_view_ids(
          'IntegerResample', _default_parameters | kwargs, view_ids)))

  return view_ids
