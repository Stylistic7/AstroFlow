import ray

from context import context
from stacks import CompositeStack, SimpleStack
from profiling import track_performance

# TODO: Ensure that all of the actions in this file are idempotent. The goal
# is to be able to apply these functions very quickly when the user quarantines
# something in the UI.

# TODO: The current design of this makes un-quarantining very difficult. Maybe
# backup the groups before the quarantine operation, so they can be undone.
# In practical terms, undoing quarantine is much less common, so it's maybe
# okay if it's a bit difficult, at least for now.


# TODO: This is supposed to run just on a single stack but it actually running
# on every stack.
@ray.remote
@track_performance
def quarantine_stacks(stack_id):
  stacks = context.grouped_stacks
  stack = stacks[stack_id]

  quarantined_stacks = context.quarantined_stacks or set()
  changed = False
  if type(stack) is CompositeStack:
    changed = any(s in quarantined_stacks for s in stack.contents)
    stack.contents = [s for s in stack.contents if s not in quarantined_stacks]
  elif type(stack) is SimpleStack:
    changed = any(s in quarantined_stacks for s in stack.dependencies.values())
    stack.dependencies = {
        t: s
        for t, s in stack.dependencies.items()
        if s not in quarantined_stacks
    }

  if not changed:
    return stack

  # If quarantining changed the stack, recompute its metadata. (This is
  # potentially slow, so don't do it unless necessary).)
  print(f'### Quarantining changed stack {stack_id}')
  if len(stack.all_frames(stacks)) == 0:
    print(f'### Stack {stack_id} is now empty and will be deleted')
    return None
  else:
    # TODO: This won't work for calibration stacks, but will people actually
    # quarantine calibration stacks?
    stack.update_metadata(stacks)
    return stack

  # Eliminate all stacks that are now empty.
  # num_stacks_before = len(stacks)
  # stacks = {
  #     stack_id: stack
  #     for stack_id, stack in stacks.items()
  #     if len(stack.all_frames(stacks)) > 0
  # }
  # if len(stacks) > num_stacks_before:
  #   print(f'### Eliminated {len(stacks) - num_stacks_before} empty stacks')

  # return stacks


# @ray.remote
# def quarantine_targets():
#   quarantined_stacks = context.quarantined_stacks or set()
#   quarantined_targets = context.quarantined_targets or set()

#   # Remove all of the quarantined targets from the dict of targets.
#   targets = {
#       target_id: target
#       for target_id, target in context.targets.items()
#       if target_id not in quarantined_targets
#   }

#   # Go through all of the remaining targets, and remove the quarantined stacks
#   # from them.
#   # TODO: Also remove any stacks that no longer exist in the set of stacks,
#   # e.g. those that became empty and were deleted.
#   for target in targets.values():
#     target.stack_ids = [
#         s for s in target.stack_ids if s not in quarantined_stacks
#     ]

#   # TODO: Delete empty targets

#   context.targets = targets

# @ray.remote
# def quarantine_mosaics():
#   quarantined_targets = context.quarantined_targets or set()
#   quarantined_mosaics = context.quarantined_mosaics or set()

#   # Remove all of the quarantined mosaics from the dict of mosaics.
#   # TODO: Also remove any targets that no longer exist in the set of targets,
#   # e.g. those that became empty and were deleted.
#   mosaics = {
#       mosaic_id: mosaic
#       for mosaic_id, mosaic in context.mosaics.items()
#       if mosaic_id not in quarantined_mosaics
#   }

#   # Go through all of the remaining mosaics, and remove the quarantined targets
#   # from them.
#   for mosaic in mosaics.values():
#     mosaic.target_ids = [
#         t for t in mosaic.target_ids if t not in quarantined_targets
#     ]

#   # TODO: Delete empty mosaics

#   context.mosaics = mosaics
