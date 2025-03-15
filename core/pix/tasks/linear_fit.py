from typing import Any, cast

from pix.classes import ShellScriptTask, Image, ImageList
from pix.context import append_task
from pix.processes import enum, execute_on_view_ids

_default_parameters = {
    'referenceViewId': '',
    'rejectLow': 0.000000,
    'rejectHigh': 0.920000,
}


def LinearFit(images: ImageList, reference_image: Image,
              **kwargs: Any) -> ImageList:
  assert all(i.view_id is not None for i in images)
  view_ids = [cast(str, i.view_id) for i in images]

  parameters = _default_parameters | kwargs | {
      'referenceViewId': reference_image.view_id
  }

  append_task(
      ShellScriptTask(
          javascript=execute_on_view_ids('LinearFit', parameters, view_ids)))

  return images
