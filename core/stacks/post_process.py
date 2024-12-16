import ray
import os
import inspect
from pathlib import Path
import shutil

import file_io
from artifacts import Artifact, FileArtifact
from context import context
from artifacts.decorators import saves_artifacts
from stacks.artifacts import stack_files_artifacts
from stacks.types import StackHash
import pix
from pix.decorators import PixInsight
from pix.tasks import BlurXTerminator, AutomaticBackgroundExtractor, SaveImages
from pix.classes import ImageList
from hashing import hash_list
from profiling import track_performance
from dependency_graph.astroflow_node import AstroFlowNode


# TODO: Find a way to disable or mock these these during tests.
@PixInsight()
def _post_process_stack(images: ImageList) -> ImageList:
  images = BlurXTerminator(images)
  images = AutomaticBackgroundExtractor(images)

  return images


@AstroFlowNode(resources={'PixInsight': 1})
def post_process_stack(master: Path) -> Path:
  print(f'### post_process_stack, master {master}')

  results = _post_process_stack([master])

  output_filename = next(iter(results)).filename
  final_filename = master.with_stem(master.stem + "_post_processed")
  shutil.copy(output_filename, final_filename)

  return final_filename
