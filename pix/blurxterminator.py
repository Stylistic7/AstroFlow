from pathlib import Path

from .classes import ShellScriptTask
from .abstract_image import AbstractImage
from .context import append_task, add_images_to_save
from .processes import enum, execute_on_view_ids

_default_parameters = {
    'ai_file': 'BlurXTerminator.4.mlpackage',
    'correct_only': False,
    'correct_first': False,
    'nonstellar_then_stellar': False,
    'lum_only': False,
    'sharpen_stars': 0.25,
    'adjust_halos': 0.00,
    'nonstellar_psf_diameter': 0.00,
    'auto_nonstellar_psf': True,
    'sharpen_nonstellar': 0.90,
}


def BlurXTerminator(images, **kwargs):
  print('### Entering BlurXTerminator')

  view_ids = [i.view_id for i in images]
  append_task(
      ShellScriptTask(javascript=execute_on_view_ids(
          'BlurXTerminator', _default_parameters | kwargs, view_ids)))

  def extend_filename(original_filename, string):
    path = Path(original_filename)
    return str(path.parent / (path.stem + string + path.suffix))

  output_images = [
      AbstractImage(filename=extend_filename(image.filename, '_blurx'),
                    view_id=image.view_id,
                    dirty=True) for image in images
  ]

  add_images_to_save(output_images)

  print(f'### BlurXTerminator returning {output_images}')
  return output_images
