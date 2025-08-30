from typing import Any, cast

from pix.classes import ShellScriptTask, Image, ImageList
from pix.context import append_task
from pix.processes import execute_on_view_ids

_default_parameters = {
    # TODO: Make this cross-platform.
    'ai_file': '/Applications/PixInsight/library/StarXTerminator.11.mlmodel',
    'stars': True,
    'unscreen': False,
    'overlap': 0.20,
}

_rename_javascript = """
var windows = ImageWindow.windows;
windows[windows.length - 1].currentView.id = "{view_id}";
"""


def StarXTerminator(images: ImageList, **kwargs: Any) -> ImageList:
  assert all(i.view_id is not None for i in images)

  output_images = []
  for image in images:
    view_id = cast(str, image.view_id)
    append_task(
        ShellScriptTask(javascript=execute_on_view_ids(
            'StarXTerminator', _default_parameters | kwargs, [view_id])))

    stars_view_id = view_id + '_stars'
    append_task(
        ShellScriptTask(javascript=_rename_javascript.format(
            view_id=stars_view_id)))

    output_images.append(
        Image(filename=image.filename,
              tag='starless',
              view_id=view_id,
              dirty=True))
    output_images.append(
        Image(filename=image.filename,
              tag='stars',
              view_id=stars_view_id,
              dirty=True))

  return ImageList(output_images)
