import tempfile
import traceback
import functools
import uuid
import sys

from . import shell_script
from . import open_images
from . import save_images
from .abstract_image import AbstractImage


def PixInsight():
  depth = 0

  def tracer(frame, event, arg):
    nonlocal depth
    if event == 'call':
      depth += 1
      if depth > 1:
        return None
      frame.f_trace_lines = False
      frame.f_trace_opcodes = False
      return tracer

    print('### f_locals', frame.f_locals)

    # Images that have a filename but no view_id are impossible.
    #
    # Images that have no filename but a view_id are a predicament,
    # because I have no idea what the filename should be. They should also
    # be impossible. When the AbstractImage is made, it should contain
    # info about what it was derived from.
    #
    # Images that have both filename and view_id are fine. If the dirty bit
    # is set, flush them to disk.
    images_to_save = []
    for name, value in frame.f_locals.items():
      if type(value) is list:
        if not all(type(x) is AbstractImage for x in value):
          print(f'### ignoring list variable {name} that does not '
                f'contain images')
          continue

      images = value
      for image in images:
        if image in images_to_save:
          continue

        if image.filename:
          assert image.view_id, f'Impossible image {image}'

        if image.view_id:
          # TODO: Create a new image filename that's an extension of the image
          # used to create this image.
          assert image.filename, f'Impossible image {image}'

        if image.dirty:
          images_to_save.append(image)

    if images_to_save:
      save_images.SaveImages(images_to_save)

    return tracer

  def decorator(func):

    @functools.wraps(func)
    def wrapper(filenames, *args, **kwargs):
      print(f'### Entering wrapper, filenames {filenames} '
            f'args {args} kwargs {kwargs}')
      if type(filenames) is not list:
        filenames = [filenames]

      func_globals = func.__globals__

      # Save copy of any global values that will be replaced.
      # saved_values = {
      #     key: func_globals[key]
      #     for key in context if key in func_globals
      # }
      try:
        with tempfile.TemporaryDirectory() as tmp_dir:
          context = dict(tasks=list(), process_number=0)

          context['tmp_dir'] = tmp_dir
          func_globals.update(context)

          # Load the given filenames and turn them into AbstractImages.
          view_ids = ['foo'] * len(filenames)
          images = open_images.OpenImages(filenames, view_ids)

          sys.settrace(tracer)
          try:
            print(f'### Calling function with tmp_dir {tmp_dir}')
            result = func(images, *args, **kwargs)
          finally:
            sys.settrace(None)

          print(
              f"### Shell script is ready, with {len(context['tasks'])} tasks")
          shell_script.run_script(context['tasks'], dry_run=True)
      except Exception as e:
        traceback.print_exc(e)

      # finally:
      #   func_globals.update(saved_values)  # Restore replaced globals.

      return result

    return wrapper

  return decorator
