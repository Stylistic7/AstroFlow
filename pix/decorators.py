import tempfile
import traceback
import functools
import os
from pathlib import Path
import inspect
from addict import Dict

from . import shell_script
from . import open_images
from . import save_images
from .file_io import read_xisf_header, add_xisf_metadata
from .hashing import hash_list, get_content_hash

_CONTENT_HASH_KEY = 'XISF:AutoPix:ContentHash'


def _extend_filename(image, funcname):
  path = Path(image.filename)

  if image.tag:
    return str(path.parent /
               (path.stem + '_' + funcname + '_' + image.tag + path.suffix))
  else:
    return str(path.parent / (path.stem + '_' + funcname + path.suffix))


# TODO: Refactor this function into pieces.
def PixInsight(dry_run=False, debug=False):

  def decorator(func):

    @functools.wraps(func)
    def wrapper(filenames, *args, **kwargs):
      print(f'### Entering wrapper, filenames {filenames} '
            f'args {args} kwargs {kwargs}')
      if type(filenames) is not list:
        filenames = [filenames]

      # The content hash of images produced by this PixInsight function
      # are composed of the hash of the function's code, the hashes of all
      # the function's input images, and the output image's view_id.
      input_image_hashes = [get_content_hash(f) for f in filenames]

      try:
        with tempfile.TemporaryDirectory() as tmp_dir:
          context = dict(tmp_dir=tmp_dir,
                         tasks=list(),
                         images_to_save=[],
                         debug_images=[],
                         process_number=0)

          def call_with_context(func, *args, **kwargs):
            func.__globals__.update(context)
            return func(*args, **kwargs)

          # Load the given filenames and their metadata.
          images = call_with_context(open_images.OpenImages, filenames)

          # Call the decorated PixInsight function.
          print('### Calling PixInsight function')
          result = call_with_context(func, images, *args, **kwargs)
          assert type(result) in [list, tuple]
          print('### PixInsight function returned correctly')

          # If the PixInsight function didn't add any tasks, there is nothing
          # to do. The OpenImages task doesn't count.
          if len(context['tasks']) == 1:
            return

          # Flatten the list of all the images returned by the decorated
          # PixInsight function.
          flattened = []
          for r in result:
            if type(r) in [list, tuple]:
              flattened.extend(list(r))
            else:
              flattened.append(r)

          assert all(type(x) is Dict for x in flattened)
          print(f'### PixInsight function wants to save '
                f'{len(flattened)} images')

          # Iterate over all the output images produced by the decorated
          # PixInsight function, and decide which ones actually need to be
          # saved to disk.
          images_to_save = []
          for image in flattened + context['images_to_save']:
            if image in images_to_save:
              continue

            assert image.view_id and image.filename, f'Impossible image {image}'

            if not image.dirty:
              continue

            image.filename = _extend_filename(image, func.__name__)
            image.content_hash = hash_list([inspect.getsource(func)] +
                                           input_image_hashes + [image.view_id])

            if os.path.exists(
                image.filename) and read_xisf_header(image.filename).get(
                    _CONTENT_HASH_KEY, None) == image.content_hash:
              continue

            print(f'### will save: {image}')
            images_to_save.append(image)

          if images_to_save:
            call_with_context(save_images.SaveImages, images_to_save)

          print(f"### Shell script is ready, with "
                f"{len(context['tasks'])} tasks")
          print(f'### images_to_save is', images_to_save)
          shell_script.run_script(context['tasks'], dry_run=dry_run)

          if not dry_run:
            for image in images_to_save:
              assert os.path.exists(image.filename)
              add_xisf_metadata(image.filename, image.filename,
                                {_CONTENT_HASH_KEY: image.content_hash})
      except Exception as e:
        traceback.print_exc(e)

      # TODO: It's weird to return this instead of the normal content.
      if debug:
        return context['debug_images']

      return result

    return wrapper

  return decorator
