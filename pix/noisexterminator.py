from .decorators import accepts_view_ids
from .classes import ShellScriptTask, ViewIdList
from .context import append_task
from .processes import enum, execute_on_view_ids

_default_parameters = {
    'ai_file': 'NoiseXTerminator.2.mlpackage',
    'denoise': 0.90,
    'detail': 0.15,
}


@accepts_view_ids
def NoiseXTerminator(view_ids, **kwargs):
  append_task(
      ShellScriptTask(javascript=execute_on_view_ids(
          'NoiseXTerminator', _default_parameters
          | kwargs, view_ids)))

  return ViewIdList(view_ids)