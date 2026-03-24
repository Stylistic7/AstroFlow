import pytest
from context import context
from stacks.composite_stack import CompositeStack


@pytest.mark.heavy
def test_orion_nebula(with_real_testdata):
    assert len(context.stacks) > 0
    assert len(context.targets) > 0

    # Every stack should have non-empty hashes.
    for stack_ref in context.stacks.values():
        stack = stack_ref.get()
        if not isinstance(stack, CompositeStack):
            continue
        assert stack.stable_hash, f"Stack has empty stable_hash"
        assert stack.content_hash, f"Stack {stack.stable_hash} has empty content_hash"

        matching_targets = [
            target_ref.get() for target_ref in context.targets.values()
            if stack.stable_hash in target_ref.get().stack_ids
        ]
        assert len(matching_targets) == 1, (
            f"Stack {stack.stable_hash} appears in {len(matching_targets)} targets"
        )
        assert matching_targets[0].stable_hash == stack.stable_hash

    # Every target ContextRef should have a content_hash, and the
    # underlying target should have valid data.
    for target_ref in context.targets.values():
        assert target_ref.content_hash, "Target ContextRef has empty content_hash"
        target = target_ref.get()
        assert target.stable_hash, "Target has empty stable_hash"
        assert len(target.stack_ids) > 0, f"Target {target.stable_hash} has no stacks"
        assert len(target.healpix_cells) > 0, f"Target {target.stable_hash} has no healpix_cells"
