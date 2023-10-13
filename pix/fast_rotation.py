from .decorators import accepts_view_ids
from .classes import ShellScriptTask, ViewIdList
from .context import append_task
from .processes import enum, execute_on_view_ids

_default_parameters = {
    'mode': enum('FastRotation.prototype.HorizontalMirror'),
    'noGUIMessages': True,
}


@accepts_view_ids
def FastRotation(view_ids, **kwargs):
  append_task(
      ShellScriptTask(
          javascript=execute_on_view_ids('FastRotation', _default_parameters
                                         | kwargs, view_ids)))

  return ViewIdList(view_ids)