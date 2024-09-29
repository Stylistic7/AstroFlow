from typing import Any, cast

from pix.classes import ShellScriptTask, ImageList
from pix.context import append_task
from pix.processes import enum, execute_on_view_ids

_default_parameters = {
    'mode': enum('FastRotation.prototype.HorizontalMirror'),
    'noGUIMessages': True,
}


def FastRotation(images: ImageList, **kwargs: Any) -> ImageList:

  assert all(i.view_id is not None for i in images)
  view_ids = [cast(str, i.view_id) for i in images]

  append_task(
      ShellScriptTask(
          javascript=execute_on_view_ids('FastRotation', _default_parameters |
                                         kwargs, view_ids)))

  return images
