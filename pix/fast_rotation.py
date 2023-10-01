from .decorators import accepts_view_ids
from .classes import ShellScriptTask, ViewIdList
from .context import append_task
from .processes import code_for_process

_default_parameters = {
    'mode': 'FastRotation.prototype.HorizontalMirror',
    'noGUIMessages': True,
}


@accepts_view_ids
def FastRotation(view_ids, **kwargs):
  append_task(
      ShellScriptTask(
          javascript=code_for_process('FastRotation', _default_parameters
                                      | kwargs, view_ids)))

  return ViewIdList(view_ids)