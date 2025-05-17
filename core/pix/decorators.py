import tempfile
import functools
import os

from typing import Callable, Any, cast

from pix.classes import PixInsightFn, FilenameList, ImageList, Image

from pix.shell_script import run_script
from pix.tasks import OpenImages, SaveImages

# Some philosophy here:
#
# There are at least two intended use cases:
#
# - A PixInsight function that is plugged into AstroFlow. AstroFlow will be
#   responsible for opening the images before calling the function, and
#   saving the images to disk after calling the function. An example of
#   this use case is:
#
#     @PixInsight()
#     def foo(images: ImageList) -> ImageList:
#       return BlurXTerminator(images)
#
#   The @PixInsight decorator will maintain a temporary directory, so the
#   user never needs to worry about where the files come from or go; they only
#   need to worry about what operations are performed.
#
# - A PixInsight function that is run independently, e.g. from a script outside
#   of AstroFlow. The user will probably pass filenames in and filenames out,
#   with intermediate steps just kept in memory in PixInsight.
#
#   An example for this use case is:
#
#     @PixInsight()
#     def foo(filenames_in: FilenameList, filenames_out: FilenameList) -> None:
#       images = OpenImages(filenames_in)
#       images = BlurXTerminator(images)
#       SaveImages(images, filenames_out)
#
# The @PixInsight decorator really only has one job: it collects the JavaScript
# and shell script code emitted by each step in the wrapped function, and
# handles dipatching code to PixInsight and waiting for completion.
#
# All of the file-opening and file-saving stuff is outside of that core job.
#
# The file-opening and file-saving code ultimately has to end up included in
# the JavaScripts and shell scripts run inside PixInsight, though, so it does
# make sense for the decorator to handle it, too. Or maybe there can be two
# distinct decorators that share a lot of code. Or maybe the decorator can
# select its behavior by matching the function's signature.


# TODO: Refactor this function into pieces.
def PixInsight(
    slots: list[int] | None = None,
    timeout: float | None = None) -> Callable[[PixInsightFn], PixInsightFn]:

  def decorator(func: PixInsightFn) -> PixInsightFn:

    @functools.wraps(func)
    def wrapper(filenames: FilenameList, *args: Any,
                **kwargs: Any) -> ImageList:
      func_name = cast(Callable, func).__name__

      filenames = [os.fspath(f) for f in filenames]

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

      def normalize_images(results: list[Image]) -> list[Image]:
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

      def call_with_context(func: Callable[..., ImageList], *args: Any,
                            **kwargs: Any) -> ImageList:
        func.__globals__.update(context)
        return func(*args, **kwargs)

      # Load the given filenames and their metadata.
      images = call_with_context(OpenImages, filenames)
      context['images'] = images

      # Call the decorated PixInsight function.
      print(f'### Calling PixInsight function {func_name}')
      results = call_with_context(func, images, *args, **kwargs)
      assert isinstance(context['tasks'], list)
      print(f'### PixInsight function {func_name} returned correctly, '
            f'{len(context["tasks"])} tasks, results {results}')

      # If the PixInsight function didn't add any tasks, there is nothing
      # to do. The OpenImages task doesn't count.
      if len(context['tasks']) == 1:
        print(f'### PixInsight function {func_name} made no tasks, '
              'fwrapper returning {results}')
        return results

      results = normalize_images(results)
      call_with_context(SaveImages, results)

      print(f"### Shell script is ready for {func_name}, with "
            f"{len(context['tasks'])} tasks")
      run_script(context['tasks'], slots=slots, timeout=timeout)

      return results

    return wrapper

  return decorator
