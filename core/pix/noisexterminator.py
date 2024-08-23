from .classes import ShellScriptTask
from .context import append_task
from .processes import enum, execute_on_view_ids

_default_parameters = {
    'ai_file': 'NoiseXTerminator.2.mlpackage',
    'denoise': 0.90,
    'detail': 0.15,
}


def NoiseXTerminator(view_ids, **kwargs):
  append_task(
      ShellScriptTask(javascript=execute_on_view_ids(
          'NoiseXTerminator', _default_parameters | kwargs, view_ids)))

  return view_ids
