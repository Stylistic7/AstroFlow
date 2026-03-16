from context import context
from stacks.composite_stack import CompositeStack


def test_orion_nebula(with_real_testdata):
    assert len(context.stacks) > 0

    # Every stack should have non-null hashes and belong to exactly one target.
    for stack in context.stacks.values():
        if not isinstance(stack, CompositeStack):
            continue
        assert stack.stable_hash is not None
        assert stack.content_hash is not None

        matching_targets = [
            t for t in context.targets.values() if stack.stable_hash in t.stack_ids
        ]
        assert len(matching_targets) == 1, (
            f"Stack {stack.stable_hash} appears in {len(matching_targets)} targets"
        )
        assert matching_targets[0].stable_hash == stack.stable_hash
