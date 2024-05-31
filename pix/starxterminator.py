from pathlib import Path

from .classes import ShellScriptTask
from .abstract_image import AbstractImage
from .context import append_task
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

  def extend_filename(original_filename, string):
    path = Path(original_filename)
    return path.parent / (path.stem + string + path.suffix)

  star_images = [
      AbstractImage(filename=extend_filename(image.filename, '_stars'),
                    view_id=image.view_id + '_stars',
                    dirty=True) for image in images
  ]

  return images, star_images
