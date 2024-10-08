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
  stacks = context.stacks
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

    stack_ids = [
        s for s in target.stack_ids
        if s in stacks and s not in quarantined_stacks
    ]
    if not stack_ids:
      continue

    target = replace(target, stack_ids=stack_ids)
    output_targets[target_id] = target

  return output_targets


@ray.remote
def quarantine_mosaics() -> dict[MosaicHash, Mosaic]:
  targets = context.targets
  mosaics = context.grouped_mosaics
  print(f'### quarantine_mosaics {len(mosaics)} grouped mosaics')
  if not mosaics:
    return {}

  quarantined_targets = context.quarantined_targets or set()
  quarantined_mosaics = context.quarantined_mosaics or set()
  print(f'### quarantined_targets {quarantined_targets}')
  print(f'### quarantined_mosaics {quarantined_mosaics}')

  output_mosaics = {}
  for mosaic_id, mosaic in mosaics.items():
    if mosaic_id in quarantined_mosaics:
      continue

    new_target_ids, new_tile_indices = [], []
    for target_id, tile_indices in zip(mosaic.target_ids, mosaic.tile_indices):
      if target_id not in targets or target_id in quarantined_targets:
        continue

      new_target_ids.append(target_id)
      new_tile_indices.append(tile_indices)

    mosaic = replace(mosaic,
                     target_ids=new_target_ids,
                     tile_indices=new_tile_indices)
    output_mosaics[mosaic_id] = mosaic

  return output_mosaics
