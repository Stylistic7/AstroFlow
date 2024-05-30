from .classes import ShellScriptTask
from .context import append_task
from .processes import enum, execute_on_view_ids

_default_parameters = {
    'mode': enum('FastRotation.prototype.HorizontalMirror'),
    'noGUIMessages': True,
}


def FastRotation(view_ids, **kwargs):
  append_task(
      ShellScriptTask(
          javascript=execute_on_view_ids('FastRotation', _default_parameters |
                                         kwargs, view_ids)))

  return view_ids
