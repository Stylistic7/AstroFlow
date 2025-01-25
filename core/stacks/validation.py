from stacks.composite_stack import CompositeStacksDict
from dependency_graph.astroflow_node import AstroFlowNode


@AstroFlowNode()
def validate_stacks(stacks: CompositeStacksDict) -> None:
  print(f'#### validate_stacks validating {len(stacks)} stacks')
  for stack_id, stack in stacks.items():
    assert stack.contents, (f'### Stack {stack} has bad '
                            f'contents {stack.contents}')
    assert stack.all_frames(), (f'### Stack {stack} has bad '
                                f'all_frames {stack.all_frames()}')
  return stacks
