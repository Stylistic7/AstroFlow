from pathlib import Path
import shutil
import os

from context import context
from pix.decorators import PixInsight
from pix.tasks import BlurXTerminator, GradientCorrection, SaveImages
from pix.classes import ImageList
from decorators import cached, makes_image, measure_performance, create_thumbnail
from customization.types import CustomizedStack
from stacks.composite_stack import CompositeStack
from stacks.stacking import stack_files
from file_io.types import XisfFilePath


@PixInsight(slots=[1])
def _post_process_stack(images: ImageList,
                        output_filenames: list[Path]) -> ImageList:
  images = BlurXTerminator(images)
  images = GradientCorrection(images)
  images = SaveImages(images, output_filenames)

  return images


def post_processed_filename(stack: CompositeStack) -> XisfFilePath:
  master = stack_files.filename_fn(stack)
  return master.with_stem(master.stem + "_post_processed")


# TODO: Do post-processing immediately after each stack has been stacked.
@makes_image(CompositeStack).as_post_processed(post_processed_filename)
@cached.remote(resources={'PixInsight': 1})
@measure_performance
@create_thumbnail
def post_process_stack(stack: CustomizedStack) -> XisfFilePath:
  master = stack_files.filename_fn(stack).resolve_at(context.output_dir)
  print(f'### post_process_stack, master {master}')

  output_filename = post_process_stack.filename_fn(stack).resolve_at(
      context.output_dir)

  if context.dangerously_accept_existing_files and os.path.exists(
      output_filename):
    print(f'### Dangerously skipping post-processing because '
          f'file {output_filename} already exists')
    return output_filename

  if not os.environ.get('MOCK_PIXINSIGHT'):
    _post_process_stack([master], [output_filename])

  return output_filename
