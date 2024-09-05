import tempfile
import traceback
import functools
import os
from pathlib import Path
import inspect

from file_io.generic import get_content_hash
from hashing import hash_list

from . import shell_script
from . import open_images
from . import save_images


# TODO: Refactor this function into pieces.
def PixInsight(dry_run=False):

  def decorator(func):

    @functools.wraps(func)
    def wrapper(filenames, *args, **kwargs):
      print(f'### Entering wrapper, filenames {filenames} '
            f'args {args} kwargs {kwargs}')
      if type(filenames) is not list:
        filenames = [filenames]

      # TODO: The content hash of images produced by this PixInsight function
      # are composed of the hash of the function's code, the hashes of all
      # the function's input images, and the output image's view_id.
      # input_image_hashes = [get_content_hash(f) for f in filenames]

      tmp_dir_obj = tempfile.TemporaryDirectory()
      context = dict(tmp_dir_obj=tmp_dir_obj,
                     tasks=[],
                     images_to_save=[],
                     debug_images=[],
                     process_number=0)

      def normalize_images(results):
        output = []
        for result in results:
          result.tmp_dir_obj = tmp_dir_obj

          if result.filename:
            result.filename = os.path.join(tmp_dir_obj.name, result.filename)
          else:
            result.filename = os.path.join(tmp_dir_obj.name,
                                           f'{result.tag}.xisf')

          output.append(result)

        return output

      def call_with_context(func, *args, **kwargs):
        func.__globals__.update(context)
        return func(*args, **kwargs)

      # Load the given filenames and their metadata.
      images = call_with_context(open_images.OpenImages, filenames)

      # Call the decorated PixInsight function.
      print(f'### Calling PixInsight function {func.__name__}')
      results = call_with_context(func, images, *args, **kwargs)
      print(f'### PixInsight function {func.__name__} returned correctly, '
            f'{len(context["tasks"])} tasks, results {results}')

      # If the PixInsight function didn't add any tasks, there is nothing
      # to do. The OpenImages task doesn't count.
      if len(context['tasks']) == 1:
        print(f'### PixInsight function {func.__name__} made no tasks, '
              'fwrapper returning {results}')
        return results

      results = normalize_images(results)
      call_with_context(save_images.SaveImages, results)

      # In dry_run mode, just touch all of the files that the tasks claim they
      # produce. Otherwise, run the script.
      if dry_run:
        for result in results:
          (Path(tmp_dir_obj.name) / result.filename).touch()
      else:
        print(f"### Shell script is ready for {func.__name__}, with "
              f"{len(context['tasks'])} tasks")
        shell_script.run_script(context['tasks'], dry_run=dry_run)

      return results

    return wrapper

  return decorator
