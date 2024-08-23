from pathlib import Path
from addict import Dict

from .classes import ShellScriptTask
from .context import append_task, add_debug_images
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

  starless_images = [
      Dict(filename=image.filename,
           tag='starless',
           view_id=image.view_id,
           dirty=True) for image in images
  ]
  add_debug_images(starless_images)

  star_images = [
      Dict(filename=image.filename,
           tag='stars',
           view_id=image.view_id + '_stars',
           dirty=True) for image in images
  ]
  add_debug_images(star_images)

  return starless_images, star_images
