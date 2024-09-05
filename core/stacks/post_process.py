import ray
import os
import inspect

import file_io
from artifacts import FileArtifact
from context import context
from artifacts.decorators import saves_artifacts
from stacks.artifacts import stack_files_artifacts
import pix
from hashing import hash_list
from stacks import stack_files_artifacts


# TODO: Find a way to disable or mock these these during tests.
@pix.PixInsight()
def _post_process_stack(image):
  image = pix.BlurXTerminator(image)
  image = pix.AutomaticBackgroundExtractor(image)

  return image


def post_process_stack_artifacts(stack_id):
  artifacts = stack_files_artifacts(stack_id)
  assert len(artifacts) == 1
  artifact = next(iter(artifacts))
  path = artifact.filename
  filename = path.with_stem(path.stem + "_post_processed")

  # TODO: Should the pix library use the artifact workflow, and compute its
  # own content hashes?
  script_hash = inspect.getsource(_post_process_stack)
  stack = context.stacks[stack_id]
  content_hash = hash_list([stack.content_hash, script_hash])
  return [FileArtifact(filename, content_hash)]


@ray.remote(resources={'PixInsight': 1}, scheduling_strategy="SPREAD")
@saves_artifacts(post_process_stack_artifacts)
def post_process_stack(stack_id):
  print(f'post_process_stack, stack_id {stack_id}')

  input_artifact = stack_files_artifacts(stack_id)[0]

  output_images = _post_process_stack(input_artifact.filename)
  print('### output_images', output_images)
  assert type(output_images) is list
  assert len(output_images) == 1
  output_image = output_images[0]
  assert type(output_image) is str

  output_artifacts = post_process_stack_artifacts(stack_id)
  assert type(output_artifacts) is list
  assert len(output_artifacts) == 1
  output_artifact = output_artifacts[0]
  assert type(output_artifact) is FileArtifact

  # TODO: Should the pix library use the artifact workflow, and compute its
  # own content hashes?
  metadata = {'CONTHASH': output_artifact.content_hash}
  file_io.xisf_file_io.add_xisf_metadata(output_image, output_artifact.filename,
                                         metadata)
  print(f'### Wrote post-processed artifact {output_artifact}')

  return output_artifacts
