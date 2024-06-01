import tempfile
import traceback
import functools
import uuid
import sys
from pathlib import Path

from . import shell_script
from . import open_images
from . import save_images
from .abstract_image import AbstractImage


def _random_view_id():
  return f'View-{uuid.uuid4()}'.replace('-', '_')


def _extend_filename(image, funcname):
  path = Path(image.filename)

  if image.tag:
    return str(path.parent /
               (path.stem + '_' + funcname + '_' + image.tag + path.suffix))
  else:
    return str(path.parent / (path.stem + '_' + funcname + path.suffix))


def PixInsight(dry_run=False):

  def decorator(func):

    @functools.wraps(func)
    def wrapper(filenames, *args, **kwargs):
      print(f'### Entering wrapper, filenames {filenames} '
            f'args {args} kwargs {kwargs}')
      if type(filenames) is not list:
        filenames = [filenames]

      # Save copy of any global values that will be replaced.
      # saved_values = {
      #     key: func_globals[key]
      #     for key in context if key in func_globals
      # }
      try:
        with tempfile.TemporaryDirectory() as tmp_dir:
          context = dict(tmp_dir=tmp_dir,
                         tasks=list(),
                         images_to_save=[],
                         process_number=0)

          def call_with_context(func, *args, **kwargs):
            func.__globals__.update(context)
            return func(*args, **kwargs)

          # Load the given filenames and turn them into AbstractImages.
          view_ids = [_random_view_id() for f in filenames]
          images = call_with_context(open_images.OpenImages, filenames,
                                     view_ids)

          result = call_with_context(func, images, *args, **kwargs)
          assert type(result) in [list, tuple]

          flattened = []
          for r in result:
            if type(r) in [list, tuple]:
              flattened.extend(list(r))
            else:
              flattened.append(r)

          assert all(type(x) is AbstractImage for x in flattened)
          print(f'### PixInsight function wants to save '
                f'{len(flattened)} images')

          images_to_save = []
          for image in flattened + context['images_to_save']:
            if image in images_to_save:
              continue

            assert image.view_id and image.filename, f'Impossible image {image}'

            if image.dirty:
              image.filename = _extend_filename(image, func.__name__)
              print(f'### will save: {image}')
              images_to_save.append(image)

          if images_to_save:
            call_with_context(save_images.SaveImages, images_to_save)

          print(
              f"### Shell script is ready, with {len(context['tasks'])} tasks")
          print(f'### image_to_save is', context['images_to_save'])
          shell_script.run_script(context['tasks'], dry_run=dry_run)
      except Exception as e:
        traceback.print_exc(e)

      # finally:
      #   func_globals.update(saved_values)  # Restore replaced globals.

      return result

    return wrapper

  return decorator
