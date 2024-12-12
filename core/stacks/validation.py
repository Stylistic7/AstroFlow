import ray

from context import context
from stacks.simple_stack import SimpleStack
from stacks.composite_stack import CompositeStack, CompositeStacksDict


@ray.remote
def validate_stacks(stacks: CompositeStacksDict) -> None:
  for stack_id, stack in stacks.items():
    # if isinstance(stack, SimpleStack):
    #   assert len(stack.frames), f'### Stack {stack} has no frames'
    # elif isinstance(stack, CompositeStack):
    assert stack.contents, (f'### Stack {stack} has bad '
                            f'contents {stack.contents}')
    assert stack.all_frames(), (f'### Stack {stack} has bad '
                                f'all_frames {stack.all_frames()}')
  return stacks
