import ray
import os
import inspect

import file_io
from artifact import FileArtifact
from context import context
from decorators import saves_artifacts
from stacks.stacking import stack_files_artifacts
import pix
from hashing import hash_list


# TODO: Find a way to disable these during tests.
@pix.PixInsight()
def _post_process_stack(image):
  image = pix.BlurXTerminator(image)
  image = pix.AutomaticBackgroundExtractor(image)

  return image


def post_process_stack_artifacts(stack_id):
  stack = context.stacks[stack_id]
  filename = os.path.join(context.output_dir, 'Stacks', stack.target, 'masters',
                          f'stack_{stack_id}_post_processed.xisf')
  script_hash = inspect.getsource(_post_process_stack)
  content_hash = hash_list([stack.content_hash, script_hash])
  return [FileArtifact(filename, content_hash)]


@ray.remote(resources={'PixInsight': 1}, scheduling_strategy="SPREAD")
@saves_artifacts(post_process_stack_artifacts)
def post_process_stack(stack_id):
  print(f'post_process_stack, stack_id {stack_id}')
  stack = context.stacks[stack_id]

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

  metadata = {'CONTHASH': output_artifact.content_hash}
  file_io.xisf_file_io.add_xisf_metadata(output_image, output_artifact.filename,
                                         metadata)
  print(f'### Wrote post-processed artifact {output_artifact}')

  return output_artifacts
