from context import context
from targets.target import TargetsDict
from decorators import cached


@cached.remote
def validate_targets(targets: TargetsDict) -> None:
    stacks = context.stacks
    for target_id, target in targets.items():
        assert all(s in stacks for s in target.stack_ids), (
            f"### Target {repr(target)} has bad stack ids"
        )

    return targets
