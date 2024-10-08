import ray

from context import context
from targets.target import TargetsDict


@ray.remote
def validate_targets(ancestor_results: list[TargetsDict]) -> None:
  print(f'### Entering validate_targets')
  targets = ancestor_results[0]

  stacks = context.stacks
  for target_id, target in targets.items():
    assert all(s in stacks for s in target.stack_ids
              ), f'### Target {repr(target)} has bad stack ids'

  return targets
