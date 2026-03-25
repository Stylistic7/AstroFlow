import ray
from dataclasses import replace
from beartype import beartype
from frozendict import frozendict

from context import context
from cache.context_ref import ContextRef
from stacks.artifacts import output_filename
from stacks.composite_stack import Stack, StacksDict
from stacks import post_process_stack_artifacts
from artifacts import artifact_exists


# TODO: Include error messages, exceptions, etc. here to inform the user.
@ray.remote
@beartype
def update_state(stacks: StacksDict) -> StacksDict:
    print("### Entering update_state")
    new_refs = StacksDict()
    for stack_id, stack_ref in stacks.items():
        stack = stack_ref.get()
        if type(stack) is not Stack:
            continue

        updated = replace(
            stack,
            state=frozendict({
                "num_frames": len(stack.all_frames()),
                "stack_files": all(
                    artifact_exists(a) for a in output_filename(stack_id)
                ),
                "post_process_stack": all(
                    artifact_exists(a) for a in post_process_stack_artifacts(stack_id)
                ),
            }),
        )
        new_refs[stack_id] = ContextRef.store(("stacks", stack_id), updated)
        print(f"### Updated stack state {stack_id} to {updated.state}")

    # Merge new refs into the existing stacks dict
    existing = context.stacks or StacksDict()
    merged = StacksDict({**existing, **new_refs})
    context.stacks = merged
    return new_refs
