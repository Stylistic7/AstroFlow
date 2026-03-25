from targets.target import TargetsDict
from customization.types import CustomizedStacksDict
from decorators import node


@node
def validate_targets(targets: TargetsDict, stacks: CustomizedStacksDict) -> None:
    for target_id, target in targets.items():
        assert all(s in stacks for s in target.stack_ids), (
            f"### Target {repr(target)} has bad stack ids"
        )

    return targets
