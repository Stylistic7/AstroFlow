from beartype import beartype

from stacks.composite_stack import CompositeStacksDict
from decorators import cached


@cached.remote
def validate_stacks(stacks: CompositeStacksDict) -> CompositeStacksDict:
    print(
        f"#### validate_stacks validating {len(stacks)} stacks "
        f"type {type(stacks)}"
    )
    for stack_id, stack in stacks.items():
        assert stack.contents, (
            f"### Stack {stack} has bad contents {stack.contents}"
        )
        assert stack.all_frames(), (
            f"### Stack {stack} has bad all_frames {stack.all_frames()}"
        )

    assert None not in stacks.values()

    return stacks
