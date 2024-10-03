import ray
import os
import inspect
from pathlib import Path

import file_io
from artifacts import Artifact, FileArtifact
from context import context
from artifacts.decorators import saves_artifacts
from stacks.artifacts import stack_files_artifacts
from stacks.types import StackHash
import pix
from pix.decorators import PixInsight
from pix.tasks import BlurXTerminator, AutomaticBackgroundExtractor
from pix.classes import ImageList
from hashing import hash_list
from profiling import track_performance


# TODO: Find a way to disable or mock these these during tests.
# TODO: Don't save two copies of the post-processed images, because they take
# up huge amounts of disk space.
@PixInsight()
def _post_process_stack(images: ImageList) -> ImageList:
  images = BlurXTerminator(images)
  images = AutomaticBackgroundExtractor(images)

  return images


def post_process_stack_artifacts(stack_id: StackHash) -> list[Artifact]:
  artifacts = stack_files_artifacts(stack_id)
  assert len(artifacts) == 1
  artifact = next(iter(artifacts))
  assert isinstance(artifact, FileArtifact)
  path = Path(artifact.filename)
  filename = path.with_stem(path.stem + "_post_processed")

  # TODO: Should the pix library use the artifact workflow, and compute its
  # own content hashes?
  script_hash = inspect.getsource(_post_process_stack)
  stack = context.stacks[stack_id]
  content_hash = hash_list([stack.content_hash, script_hash])
  return [FileArtifact(str(filename), content_hash)]


@ray.remote(resources={'PixInsight': 1}, scheduling_strategy="SPREAD")
@saves_artifacts(post_process_stack_artifacts)
@track_performance
def post_process_stack(stack_id: StackHash) -> list[Artifact]:
  print(f'post_process_stack, stack_id {stack_id}')

  input_artifact = stack_files_artifacts(stack_id)[0]

  output_images = _post_process_stack([input_artifact.filename])
  output_filename = next(iter(output_images)).filename

  output_artifacts = post_process_stack_artifacts(stack_id)
  output_artifact = next(iter(output_artifacts))

  # TODO: Should the pix library use the artifact workflow, and compute its
  # own content hashes?
  metadata = {'CONTHASH': output_artifact.content_hash}
  assert isinstance(output_artifact, FileArtifact)
  file_io.xisf_file_io.add_xisf_metadata(output_filename,
                                         output_artifact.filename, metadata)
  print(f'### Wrote post-processed artifact {output_artifact}')

  return output_artifacts
