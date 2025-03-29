from pathlib import Path
import shutil
import os

from context import context
from pix.decorators import PixInsight
from pix.tasks import BlurXTerminator, GradientCorrection
from pix.classes import ImageList
from task.astroflow_node import task
from customization.types import CustomizedStack
from stacks.artifacts import output_filename


@PixInsight(slots=[1])
def _post_process_stack(images: ImageList) -> ImageList:
  images = BlurXTerminator(images)
  images = GradientCorrection(images)

  return images


@task.remote(resources={'PixInsight': 1})
def post_process_stack(stack: CustomizedStack) -> Path:
  master = output_filename(stack)
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

  return Path(final_filename)
