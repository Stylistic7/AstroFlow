import ray
from pathlib import Path
import shutil

from context import context
from pix.decorators import PixInsight
from pix.tasks import BlurXTerminator, AutomaticBackgroundExtractor
from pix.classes import ImageList
from dependency_graph.astroflow_node import AstroFlowNode


# TODO: Find a way to disable or mock these these during tests.
@PixInsight()
def _post_process_stack(images: ImageList) -> ImageList:
  images = BlurXTerminator(images)
  images = AutomaticBackgroundExtractor(images)

  return images


@AstroFlowNode(cache_result=True, resources={'PixInsight': 1})
def post_process_stack(master: Path) -> Path:
  print(f'### post_process_stack, master {master}')

  if context.mock_pixinsight:
    output_filename = master
  else:
    results = _post_process_stack([master])
    output_filename = next(iter(results)).filename

  final_filename = master.with_stem(master.stem + "_post_processed")
  shutil.copy(output_filename, final_filename)

  return Path(final_filename)
