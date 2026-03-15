import ray
from dataclasses import replace
from beartype import beartype

from context import context
from stacks.artifacts import output_filename
from stacks.composite_stack import CompositeStack, CompositeStacksDict
from stacks import post_process_stack_artifacts
from artifacts import artifact_exists


# TODO: Include error messages, exceptions, etc. here to inform the user.
@ray.remote
@beartype
def update_state(stacks: CompositeStacksDict) -> CompositeStacksDict:
    print("### Entering update_state")
    output_stacks = CompositeStacksDict()
    for stack_id, stack in stacks.items():
        if type(stack) is not CompositeStack:
            continue

        output_stacks[stack_id] = replace(
            stack,
            state={
                "num_frames": len(stack.all_frames()),
                "stack_files": all(
                    artifact_exists(a) for a in output_filename(stack_id)
                ),
                "post_process_stack": all(
                    artifact_exists(a) for a in post_process_stack_artifacts(stack_id)
                ),
            },
        )
        print(f"### Updated stack state {stack_id} to {stack.state}")

    context.or_equals("stacks", output_stacks)
    return output_stacks
