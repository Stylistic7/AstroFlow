from pathlib import Path
import os

from context import context
from pix.decorators import PixInsight
from pix.tasks import BlurXTerminator, GradientCorrection, SaveImages
from pix.classes import ImageList
from decorators import cached, measure_performance, create_thumbnail
from customization.types import CustomizedStack
from stacks.composite_stack import CompositeStack
from stacks.stacking import stack_files
from file_io.types import XisfFilePath
from file_io.generic import copy_image
from products import makes_image
from previews.types import QualityLevel


@PixInsight(slots=[1])
def _post_process_stack(images: ImageList,
                        output_filenames: list[Path]) -> ImageList:
  images = BlurXTerminator(images)
  images = GradientCorrection(images)
  SaveImages(images, output_filenames)


def post_processed_filename(stack: CompositeStack) -> XisfFilePath:
  master = stack_files.filename_fn(stack)
  return master.with_stem(master.stem + "_post_processed")


# TODO: Do post-processing immediately after each stack has been stacked.
@makes_image(CompositeStack).as_post_processed(post_processed_filename)
@cached.remote(resources={'PixInsight': 1})
@measure_performance
@create_thumbnail
def post_process_stack(mode: QualityLevel,
                       stack: CustomizedStack) -> XisfFilePath:
  master = stack_files.filename_fn(stack)
  print(f'### post_process_stack, master {master}')

  output_filename = post_process_stack.filename_fn(stack)

  if context.dangerously_accept_existing_files and os.path.exists(
      output_filename):
    print(f'### Dangerously skipping post-processing because '
          f'file {output_filename} already exists')
    return output_filename

  if context.get('mock_pixinsight', False):
    copy_image(master, output_filename)
  else:
    _post_process_stack([master], [output_filename])

  return XisfFilePath(output_filename)
