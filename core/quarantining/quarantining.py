import ray
from dataclasses import replace, asdict
import time

from context import context
from task.astroflow_node import task
from stacks.composite_stack import CompositeStack
from stacks.simple_stack import SimpleStacksDict
from attributes import ImagingAttributes
from stacks.types import StackHash


@task.remote
def quarantine_calibration_stacks(stacks: SimpleStacksDict) -> SimpleStacksDict:
  quarantined_calibration_stacks = context.quarantined_calibration_stacks or []
  return SimpleStacksDict({
      h: s for h, s in stacks.items() if h not in quarantined_calibration_stacks
  })


# TODO: Ensure that all of the actions in this file are idempotent. The goal
# is to be able to apply these functions very quickly when the user quarantines
# something in the UI.
def apply_quarantine(stack: CompositeStack) -> CompositeStack:
  quarantined_stacks = context.quarantined_stacks or []

  new_contents = {
      h: s for h, s in stack.contents.items() if h not in quarantined_stacks
  }

  for substack_id, substack in new_contents.items():
    new_dependencies = {
        t: s
        for t, s in substack.dependencies.items()
        if s not in quarantined_stacks
    }
    new_contents[substack_id] = replace(substack, dependencies=new_dependencies)

  all_frames = []
  for substack in new_contents.values():
    all_frames.extend(substack.frames)

  if not all_frames:
    return None

  attributes = asdict(
      ImagingAttributes.from_frames(all_frames, stack.all_metadata()))
  return replace(stack, contents=new_contents, **attributes)
