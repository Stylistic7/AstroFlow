import ray

from dataclasses import replace

from context import context
from stacks.simple_stack import SimpleStack
from stacks.composite_stack import CompositeStack
from targets.types import TargetHash
from targets.target import Target
from mosaics.types import MosaicHash
from mosaics.mosaic import Mosaic


# TODO: These don't belong in the compute_graph.py file
# TODO: Maybe SimpleStacks and CompositeStacks should be in separate dicts.
@ray.remote
def quarantine_stacks() -> dict[str, SimpleStack | CompositeStack]:
  stacks = context.grouped_stacks
  if not stacks:
    return {}

  quarantined_stacks = context.quarantined_stacks or set()
  return {
      stack_id: stack
      for stack_id, stack in stacks.items()
      if stack_id not in quarantined_stacks
  }


@ray.remote
def quarantine_targets() -> dict[TargetHash, Target]:
  targets = context.grouped_targets
  print(f'### quarantine_targets {len(targets)} grouped targets')
  if not targets:
    return {}

  quarantined_stacks = context.quarantined_stacks or set()
  quarantined_targets = context.quarantined_targets or set()

  output_targets = {}
  for target_id, target in targets.items():
    if target_id in quarantined_targets:
      continue

    stack_ids = [s for s in target.stack_ids if s not in quarantined_stacks]
    if not stack_ids:
      continue

    target = replace(target, stack_ids=stack_ids)
    output_targets[target_id] = target

  return output_targets


@ray.remote
def quarantine_mosaics() -> dict[MosaicHash, Mosaic]:
  mosaics = context.grouped_mosaics
  print(f'### quarantine_targets {len(mosaics)} grouped targets')
  if not mosaics:
    return {}

  quarantined_mosaics = context.quarantined_mosaics or set()
  return {
      mosaic_id: mosaic
      for mosaic_id, mosaic in mosaics.items()
      if mosaic_id not in quarantined_mosaics
  }
