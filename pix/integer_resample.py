from .decorators import accepts_view_ids
from .classes import ShellScriptTask, ViewIdList
from .context import append_task
from .processes import code_for_process

_default_parameters = {
    'zoomFactor': -2,
    'downsamplingMode': 'IntegerResample.prototype.Average',
    'xResolution': 72.000,
    'yResolution': 72.000,
    'metric': False,
    'forceResolution': False,
    'gammaCorrection': False,
    'noGUIMessages': True,
}


@accepts_view_ids
def IntegerResample(view_ids, **kwargs):
  append_task(
      ShellScriptTask(
          javascript=code_for_process('IntegerResample', _default_parameters
                                      | kwargs, view_ids)))

  return ViewIdList(view_ids)