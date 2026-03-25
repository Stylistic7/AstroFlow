from targets.target import TargetsDict
from stacks.composite_stack import CompositeStacksDict
from decorators import node


@node
def validate_targets(targets: TargetsDict, stacks: CompositeStacksDict) -> None:
    for target_id, target in targets.items():
        assert all(s in stacks for s in target.stack_ids), (
            f"### Target {repr(target)} has bad stack ids"
        )

    return targets
