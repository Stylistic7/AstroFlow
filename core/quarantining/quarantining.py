import ray
from dataclasses import replace, asdict
import time
import os

from context import context
from stacks.composite_stack import CompositeStack
from stacks.simple_stack import SimpleStack
from stacks.imaging_attributes import ImagingAttributes
from stacks.types import StackHash
from profiling import track_performance
from metadata.classes import MetadataDict

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


# TODO: Pass the stack through directly, rather than the stack_id. This
# is a better separation of concerns.
@ray.remote
@track_performance
def apply_quarantine_to_stack(
    stack_ids: list[StackHash]) -> list[CompositeStack]:
  print(f'### apply_quarantine_to_stack running on stack_ids {stack_ids}')
  stacks = context.grouped_stacks

  start_time = time.time()
  metadata = context.metadata
  end_time = time.time()
  elapsed_time = end_time - start_time
  print(f'### metadata = context.metadata took {elapsed_time:.3f}')

  output = []
  for stack_id in stack_ids:
    stack = stacks[stack_id]

    print(f'### apply_quarantine_to_stack for stack {stack}')
    assert isinstance(
        stack, CompositeStack
    ), f'### Object {stack} of type {type(stack)} is not CompositeStack'

    quarantined_stacks = context.quarantined_stacks or set()

    new_contents = {
        h: s for h, s in stack.contents.items() if h not in quarantined_stacks
    }

    for substack_id, substack in new_contents.items():
      new_dependencies = {
          t: s
          for t, s in substack.dependencies.items()
          if s not in quarantined_stacks
      }
      new_contents[substack_id] = replace(substack,
                                          dependencies=new_dependencies)

    all_frames = []
    for substack in new_contents.values():
      all_frames.extend(substack.frames)

    if not all_frames:
      continue

    start_time = time.time()
    attributes = asdict(ImagingAttributes.from_frames(all_frames, metadata))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'### ImagingAttributes.from_frames took {elapsed_time:.3f}')

    output.append(replace(stack, contents=new_contents, **attributes))

  print(f'### apply_quarantine_to_stack returning {len(output)} stacks')
  return output
