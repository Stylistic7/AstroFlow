import ray
import os

from dataclasses import replace
from typing import Any

from dependency_graph.astroflow_node import AstroFlowNode
from context import context
from targets.types import TargetHash
from stacks.composite_stack import CompositeStack, CompositeStacksDict, SimpleStacksDict
from mosaics.types import MosaicHash
from targets.target import Target, TargetsDict
from mosaics.mosaic import Mosaic, MosaicsDict


@AstroFlowNode(cache_result=False)
def quarantine_stacks() -> SimpleStacksDict | CompositeStacksDict:
  stacks = context.stacks
  if not stacks:
    return {}

  quarantined_stacks = context.quarantined_stacks or set()
  return {
      stack_id: stack
      for stack_id, stack in stacks.items()
      if stack_id not in quarantined_stacks
  }


# TODO: This should be made functional.
@AstroFlowNode(cache_result=False)
def quarantine_targets() -> dict[TargetHash, Target]:
  stacks = context.stacks
  targets = context.targets
  print(f'### quarantine_targets {len(targets)} targets')
  if not targets:
    return {}

  quarantined_stacks = context.quarantined_stacks or set()
  quarantined_targets = context.quarantined_targets or set()

  output_targets = {}
  for target_id, target in targets.items():
    if target_id in quarantined_targets:
      print(f'### removing quarantined target {target_id}')
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


@AstroFlowNode(cache_result=False)
def quarantine_mosaics() -> dict[MosaicHash, Mosaic]:
  targets = context.targets
  mosaics = context.mosaics
  print(f'### quarantine_mosaics {len(mosaics)} mosaics')
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


@AstroFlowNode()
def gather_stack_user_settings(
    *stacks_in: list[CompositeStack]) -> CompositeStacksDict:
  stacks = [item for sublist in stacks_in for item in sublist]
  stacks = {a.stable_hash: a for a in stacks}
  print(f'### gather_stack_user_settings stack ids are {stacks.keys()}')
  return stacks


@AstroFlowNode(cache_result=False)
def store_calibration_stacks(calibration_stacks: SimpleStacksDict) -> None:
  context.calibration_stacks = calibration_stacks
  print(f'### store_calibration_stacks: Stored '
        f'{len(calibration_stacks)} calibration stacks; '
        f'total is now {len(context.calibration_stacks)}')
  return calibration_stacks


@AstroFlowNode(cache_result=False)
def store_stacks(stacks: CompositeStacksDict) -> None:
  context.grouped_stacks = stacks
  print(f'### store_stacks: Stored '
        f'{len(stacks)} stacks; total is now {len(context.grouped_stacks)}')
  return stacks


@AstroFlowNode(cache_result=False)
def store_targets(targets: TargetsDict) -> None:
  context.grouped_targets = targets
  print(f'### store_targets: Stored {len(targets)} targets; '
        f'total is now {len(context.grouped_targets)}')
  return targets


@AstroFlowNode(cache_result=False)
def store_mosaics(mosaics: MosaicsDict) -> None:
  context.grouped_mosaics = mosaics
  print(f'### store_mosaics: Stored {len(mosaics)} mosaics; '
        f'total is now {len(context.grouped_mosaics)}')
  return mosaics
