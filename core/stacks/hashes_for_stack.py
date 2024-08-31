from context import context
from stacks import SimpleStack, CompositeStack, post_process_stack_artifacts
from artifacts import artifact_exists


# TODO: This needs a LOT more careful testing.
def hashes_for_stack(stack_id):
  stacks = context.stacks
  stack = stacks[stack_id]

  # Start with all of the stacks included in this stack.
  hashes = stack.all_frames(include_dependencies=True)

  # TODO: Settings need to be applied in some unified way.
  settings_key = ('stack_settings', stack_id)

  if type(stack) is SimpleStack:
    settings = context[settings_key] or {}
    dependencies = stack.dependencies | settings
  else:
    dependencies = {}

  # Include all the frames from all of the stacks' dependencies (which are
  # typically calibration stacks).
  if 'bias' in dependencies:
    hashes += stacks[dependencies['bias']].frames
  if 'dark' in dependencies:
    hashes += stacks[dependencies['dark']].frames

  # Include every known flat calibration stack, just for completeness.
  for s in stacks.values():
    if (type(s) is SimpleStack and s.image_type == 'flat' and
        s.camera == stack.camera and s.filter == stack.filter):
      hashes += s.frames

  if type(stack) is CompositeStack:
    for artifact in post_process_stack_artifacts(stack_id):
      if artifact_exists(artifact):
        hashes.append(artifact.content_hash)

  return hashes
