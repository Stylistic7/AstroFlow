from typing import Any, cast

from pix.classes import ShellScriptTask, Image, ImageList
from pix.context import append_task, add_images_to_save
from pix.processes import execute_on_view_ids

_default_parameters = {
    # TODO: Make this cross-platform.
    'ai_file': '/Applications/PixInsight/library/StarXTerminator.11.mlmodel',
    'stars': True,
    'unscreen': False,
    'overlap': 0.20,
}


def StarXTerminator(images: ImageList, **kwargs: Any) -> ImageList:
  assert all(i.view_id is not None for i in images)
  view_ids = [cast(str, i.view_id) for i in images]

  append_task(
      ShellScriptTask(javascript=execute_on_view_ids(
          'StarXTerminator', _default_parameters | kwargs, view_ids)))

  output_images = []
  for image in images:
    output_images.append(
        Image(filename=image.filename,
              tag='stars',
              view_id=image.view_id,
              dirty=True))
    output_images.append(
        Image(filename=image.filename,
              tag='starless',
              view_id=image.view_id,
              dirty=True))

  add_images_to_save(output_images)

  return output_images
