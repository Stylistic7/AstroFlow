from context import context
from stacks.composite_stack import CompositeStack


def test_orion_nebula(with_real_testdata):
    assert len(context.stacks) > 0

    # Every stack should have non-null hashes and belong to exactly one target.
    for stack_ref in context.stacks.values():
        stack = stack_ref.get()
        if not isinstance(stack, CompositeStack):
            continue
        assert stack.stable_hash is not None
        assert stack.content_hash is not None

        matching_targets = [
            target_ref.get() for target_ref in context.targets.values()
            if stack.stable_hash in target_ref.get().stack_ids
        ]
        assert len(matching_targets) == 1, (
            f"Stack {stack.stable_hash} appears in {len(matching_targets)} targets"
        )
        assert matching_targets[0].stable_hash == stack.stable_hash
