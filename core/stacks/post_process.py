from pathlib import Path
import shutil
import os

from context import context
from pix.decorators import PixInsight
from pix.tasks import BlurXTerminator, GradientCorrection
from pix.classes import ImageList
from decorators import cached, register_image_producer, measure_performance, create_thumbnail
from customization.types import CustomizedStack
from stacks.composite_stack import CompositeStack
from stacks.stacking import stack_files
from file_io.types import XisfFilePath
from previews.types import QualityLevel


@PixInsight(slots=[1])
def _post_process_stack(images: ImageList) -> ImageList:
  images = BlurXTerminator(images)
  images = GradientCorrection(images)

  return images


def post_processed_filename(stack: CompositeStack) -> XisfFilePath:
  master = stack_files.filename_fn(stack)
  return master.with_stem(master.stem + "_post_processed")


# TODO: Do post-processing immediately after each stack has been stacked.
@register_image_producer(CompositeStack, QualityLevel.POST_PROCESSED,
                         post_processed_filename)
@cached.remote(resources={'PixInsight': 1})
@measure_performance
@create_thumbnail
def post_process_stack(stack: CustomizedStack) -> Path:
  master = context.output_dir / stack_files.filename_fn(stack)
  print(f'### post_process_stack, master {master}')

  final_filename = post_process_stack.filename_fn(stack)

  if context.dangerously_accept_existing_files and os.path.exists(
      final_filename):
    print(f'### Dangerously skipping post-processing because '
          f'file {final_filename} already exists')
    return Path(final_filename)

  if os.environ.get('MOCK_PIXINSIGHT'):
    post_processed = master
  else:
    results = _post_process_stack([master])
    post_processed = next(iter(results)).filename

  shutil.copy(post_processed, final_filename)
  return XisfFilePath(final_filename)
