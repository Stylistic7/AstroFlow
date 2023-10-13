from .decorators import accepts_view_ids
from .classes import ShellScriptTask, ViewIdList
from .context import append_task
from .processes import execute_on_view_ids

_default_parameters = {
    # TODO: Make this cross-platform.
    'ai_file': '/Applications/PixInsight/library/StarXTerminator.11.mlmodel',
    'stars': True,
    'unscreen': True,
    'overlap': 0.50,
}


@accepts_view_ids
def StarXTerminator(view_ids, **kwargs):
  append_task(
      ShellScriptTask(
          javascript=execute_on_view_ids('StarXTerminator', _default_parameters
                                         | kwargs, view_ids)))

  output_view_ids = [v + '_stars' for v in view_ids]
  return ViewIdList(output_view_ids)