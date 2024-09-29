import ray
from dataclasses import replace
from typing import Any

from context import context
from stacks.composite_stack import CompositeStack, CompositeStacksDict
from stacks import stack_files_artifacts, post_process_stack_artifacts
from artifacts import artifact_exists
from profiling import track_performance


# TODO: Include error messages, exceptions, etc. here to inform the user.
@ray.remote
@track_performance
def update_state(ancestor_results: list[Any]) -> CompositeStacksDict:
  print('### Entering update_state')
  stacks = ancestor_results[0]

  output_stacks = CompositeStacksDict()
  for stack_id, stack in stacks.items():
    if type(stack) is not CompositeStack:
      continue

    output_stacks[stack_id] = replace(
        stack,
        state={
            'num_frames':
                len(stack.all_frames()),
            'stack_files':
                all(
                    artifact_exists(a) for a in stack_files_artifacts(stack_id)
                ),
            'post_process_stack':
                all(
                    artifact_exists(a)
                    for a in post_process_stack_artifacts(stack_id))
        })
    print(f'### Updated stack state {stack_id} to {stack.state}')

  context.or_equals('stacks', output_stacks)
  return output_stacks
