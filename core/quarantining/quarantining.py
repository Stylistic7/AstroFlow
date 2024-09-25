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


# TODO: Pass the stack through directly, rather than the stack_id. This
# is a better separation of concerns.
@ray.remote
@track_performance
def apply_quarantine_to_stack(stack_id):
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
