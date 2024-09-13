import ray

from context import context
from stacks import CompositeStack, SimpleStack

# TODO: Ensure that all of the actions in this file are idempotent. The goal
# is to be able to apply these functions very quickly when the user quarantines
# something in the UI.

# TODO: The current design of this makes un-quarantining very difficult. Maybe
# backup the groups before the quarantine operation, so they can be undone.
# In practical terms, undoing quarantine is much less common, so it's maybe
# okay if it's a bit difficult, at least for now.


@ray.remote
def quarantine_stacks():
  # Remove all of the quarantined stacks from the dict of stacks.
  quarantined_stacks = context.quarantined_stacks or set()
  stacks = {
      stack_id: stack
      for stack_id, stack in context.stacks.items()
      if stack_id not in quarantined_stacks
  }

  # Go through all of the remaining stacks, and remove the quarantined stacks
  # from `contents` and `dependencies`.
  for stack_id, stack in stacks.items():
    changed = False

    if type(stack) is CompositeStack:
      changed |= any(s in stack.contents for s in quarantined_stacks)
      stack.contents = [
          s for s in stack.contents if s not in quarantined_stacks
      ]
    elif type(stack) is SimpleStack:
      changed |= any(
          s in stack.dependencies.values() for s in quarantined_stacks)
      stack.dependencies = {
          t: s
          for t, s in stack.dependencies.items()
          if s not in quarantined_stacks
      }

    # If quarantining changed the stack, recompute its metadata.
    if changed:
      print(f'### Quarantining changed stack {stack_id}')
      all_frames = stack.all_frames()
      if len(all_frames) == 0:
        print(f'### Stack {stack_id} is now empty and will be deleted')
      else:
        # TODO: This won't work for calibration stacks, but will people actually
        # quarantine calibration stacks?
        stack.update_metadata()

  # Eliminate all stacks that are now empty.
  stacks = {
      stack_id: stack
      for stack_id, stack in stacks.items()
      if len(stack.all_frames()) > 0
  }

  context.stacks = stacks


@ray.remote
def quarantine_targets():
  quarantined_stacks = context.quarantined_stacks or set()
  quarantined_targets = context.quarantined_targets or set()

  # Remove all of the quarantined targets from the dict of targets.
  targets = {
      target_id: target
      for target_id, target in context.targets.items()
      if target_id not in quarantined_targets
  }

  # Go through all of the remaining targets, and remove the quarantined stacks
  # from them.
  for target in targets.values():
    target.stack_ids = [
        s for s in target.stack_ids if s not in quarantined_stacks
    ]

  context.targets = targets


@ray.remote
def quarantine_mosaics():
  quarantined_targets = context.quarantined_targets or set()
  quarantined_mosaics = context.quarantined_mosaics or set()

  # Remove all of the quarantined mosaics from the dict of mosaics.
  mosaics = {
      mosaic_id: mosaic
      for mosaic_id, mosaic in context.mosaics.items()
      if mosaic_id not in quarantined_mosaics
  }

  # Go through all of the remaining mosaics, and remove the quarantined targets
  # from them.
  for mosaic in mosaics.values():
    mosaic.target_ids = [
        t for t in mosaic.target_ids if t not in quarantined_targets
    ]

  context.mosaics = mosaics
