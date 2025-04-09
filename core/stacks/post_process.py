from pathlib import Path
import shutil
import os

from context import context
from pix.decorators import PixInsight
from pix.tasks import BlurXTerminator, GradientCorrection
from pix.classes import ImageList
from task.astroflow_node import task
from customization.types import CustomizedStack
from stacks.composite_stack import CompositeStack
from stacks.artifacts import output_filename
from stacks.stacking import stack_files
from file_io.types import XisfFilePath
from products.register import register_image_producer
from previews import QualityLevel


@PixInsight(slots=[1])
def _post_process_stack(images: ImageList) -> ImageList:
  images = BlurXTerminator(images)
  images = GradientCorrection(images)

  return images


# TODO: Do post-processing immediately after each stack has been stacked.
@register_image_producer(CompositeStack, QualityLevel.POST_PROCESSED,
                         output_filename)
@task.remote(resources={'PixInsight': 1})
def post_process_stack(stack: CustomizedStack) -> Path:
  master = context.output_dir / stack_files.filename_fn(stack)
  print(f'### post_process_stack, master {master}')

  final_filename = master.with_stem(master.stem + "_post_processed")

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
