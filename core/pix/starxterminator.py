from pathlib import Path

from .classes import ShellScriptTask, Image
from .context import append_task, add_images_to_save
from .processes import execute_on_view_ids

_default_parameters = {
    # TODO: Make this cross-platform.
    'ai_file': '/Applications/PixInsight/library/StarXTerminator.11.mlmodel',
    'stars': True,
    'unscreen': True,
    'overlap': 0.50,
}


def StarXTerminator(images, **kwargs):
  view_ids = [i.view_id for i in images]
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
