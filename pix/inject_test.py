from functools import wraps
import os
import inspect
import tempfile

import shell_script
from image_solver import ImageSolver
from blurxterminator import BlurXTerminator


def PixInsight():
  context = dict(tasks=[], process_number=0)
  """ Decorator factory. """
  def variable_injector(func):
    """ Decorator. """
    @wraps(func)
    def decorator(*args, **kwargs):
      func_globals = func.__globals__

      # Save copy of any global values that will be replaced.
      saved_values = {
          key: func_globals[key]
          for key in context if key in func_globals
      }
      try:
        with tempfile.TemporaryDirectory() as tmp_dir:
          print(f'### Calling function with tmp_dir {tmp_dir}')
          context['tmp_dir'] = tmp_dir
          func_globals.update(context)

          result = func(*args, **kwargs)
          print(
              f"### Shell script is ready, with {len(context['tasks'])} tasks")
          for task in context['tasks']:
            print(f'### task {task}')

          shell_script.run_script(context['tasks'])
      finally:
        func_globals.update(saved_values)  # Restore replaced globals.

      return result

    return decorator

  return variable_injector


# TODO: Remove the () if possible.
#
# TODO: images eventually needs to be a class that has accessors like:
# rgb = [images.R, images.G, images.B]
#
# TODO: This code currently literally passes the output of one function to
# the next, but we need to look at its type and potentially add types in between.
@PixInsight()
def post_process_target(images):
  images = ImageSolver(images)
  images = BlurXTerminator(images)


if __name__ == '__main__':
  shell_script._clear_cache()

  filenames = [
      "/Users/nobody/Projects/Astro/Processed/Filamentary nebula/masters/master_light_Filamentary nebula_C11_H.xisf",
      "/Users/nobody/Projects/Astro/Processed/Filamentary nebula/masters/master_light_Filamentary nebula_RC71_O.xisf"
  ]
  post_process_target(shell_script.FileList(filenames))
