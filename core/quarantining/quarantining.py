import ray
from dataclasses import replace

from context import context
from stacks import CompositeStack, SimpleStack, FrameHash, ImagingAttributes
from profiling import track_performance

# TODO: Ensure that all of the actions in this file are idempotent. The goal
# is to be able to apply these functions very quickly when the user quarantines
# something in the UI.

# TODO: The current design of this makes un-quarantining very difficult. Maybe
# backup the groups before the quarantine operation, so they can be undone.
# In practical terms, undoing quarantine is much less common, so it's maybe
# okay if it's a bit difficult, at least for now.


def _apply_to_simple_stack(stack: SimpleStack) -> SimpleStack:
  quarantined_stacks = context.quarantined_stacks or set()

  new_dependencies = {
      t: s for t, s in stack.dependencies.items() if s not in quarantined_stacks
  }
  if new_dependencies == stack.dependencies:
    return stack

  return replace(stack, dependencies=new_dependencies)


def _apply_to_composite_stack(stack: CompositeStack) -> CompositeStack:
  stacks = context.grouped_stacks
  quarantined_stacks = context.quarantined_stacks or set()

  new_contents = {
      h: s for h, s in stack.contents.items() if h not in quarantined_stacks
  }
  if new_contents == stack.contents:
    return stack

  all_frames = []
  for substack in stack.contents.values():
    all_frames.extend(substack.frames)

  attributes = ImagingAttributes.from_frames(all_frames) | {
      'contents': new_contents
  }

  return replace(stack, **attributes)


# TODO: Pass the stack through directly, rather than the stack_id. This
# is a better separation of concerns.
@ray.remote
@track_performance
def apply_quarantine_to_stack(stack_id):
  stacks = context.grouped_stacks
  stack = stacks[stack_id]

  if type(stack) is CompositeStack:
    return _apply_to_composite_stack(stack)
  elif type(stack) is SimpleStack:
    return _apply_to_simple_stack(stack)
