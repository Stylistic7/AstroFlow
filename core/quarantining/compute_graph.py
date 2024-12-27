import ray
import os

from dataclasses import replace
from typing import Any

from dependency_graph.astroflow_node import AstroFlowNode
from context import context, WriteMode
from calibration import associate_with_stack
from targets.validation import validate_targets
from targets.types import TargetHash
from dependency_graph import Graph, make_graph_from_list
from quarantining import apply_quarantine_to_stack
from quarantining.subframe_selection import select_subframes
from stacks.simple_stack import SimpleStack
from stacks.composite_stack import CompositeStack, CompositeStacksDict
from stacks.types import StackHash
from stacks.validation import validate_stacks
from mosaics.types import MosaicHash
from mosaics.validation import validate_mosaics
from targets.target import Target, TargetsDict
from mosaics.mosaic import Mosaic, MosaicsDict


# TODO: Maybe SimpleStacks and CompositeStacks should be in separate dicts.
@AstroFlowNode()
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


@AstroFlowNode()
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


@AstroFlowNode()
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


@AstroFlowNode()
def apply_settings(stacks: list[CompositeStack]) -> list[CompositeStack]:
  print(f'### apply_settings running on {len(stacks)} stacks')
  output = []
  for stack in stacks:
    if stack is None:
      continue

    print(f'### apply_settings for stack {stack}')
    assert isinstance(stack,
                      CompositeStack), f'Object {stack} is not a CompositeStack'

    new_contents = {}
    for substack in stack.contents.values():
      settings_key = ('stack_settings', substack.content_hash)
      settings = context[settings_key] or {}
      dependencies = substack.dependencies | settings
      new_substack = replace(substack, dependencies=dependencies)
      new_contents[new_substack.content_hash] = new_substack

    output.append(replace(stack, contents=new_contents))

  print(f'### apply_settings returning {len(output)} stacks')
  return output


# This graph applies user settings to a single stack in the `grouped_stacks`
# dict, passing it through graph arguments. If the stack is quarantined, or
# becomes empty after subframe selection, it will not make it into the final
# `stacks` dict.
#
# `apply_quarantine_to_stack()` is applied to a single Stack, and either returns
# that Stack, modified, or None.
#
# `select_subframes()` is applied to the single Stack (or None), and either
# returns that Stack, modified, or None.
#
# `associate_with_stack()` is applied to the single Stack (or None), and returns
# a dict containing the Stack, plus the calibration Stacks that it depends on,
# or None.
#
# TODO: This is very slow right now, despite running in parallel. These
# functions all retrieve metadata from the context, which takes more than a
# second:
#
# apply_quarantine_to_stack()
# select_subframes()
#
# Maybe the best thing to do is pass the metadata ref around?
@AstroFlowNode()
def build_stack_user_settings_subgraph(stack_ids: list[StackHash]) -> Graph:
  print(f'### build_stack_user_settings_subgraph stack_ids {stack_ids}')
  return make_graph_from_list([(apply_quarantine_to_stack, stack_ids),
                               select_subframes, associate_with_stack,
                               apply_settings])


@AstroFlowNode()
def gather_stack_user_settings(
    *stacks_in: list[CompositeStack]) -> CompositeStacksDict:
  stacks = [item for sublist in stacks_in for item in sublist]
  stacks = {a.stable_hash: a for a in stacks}
  print(f'### gather_stack_user_settings stack ids are {stacks.keys()}')
  context.stacks = stacks
  print(f'### gather_stack_user_settings: Stored {len(context.stacks)} stacks')

  return stacks


@AstroFlowNode()
def store_calibration_stacks() -> None:
  calibration_stacks = context.calibration_stacks
  context.or_equals('stacks', calibration_stacks)
  print(f'### store_calibration_stacks: Stored '
        f'{len(calibration_stacks)} calibration stacks')


@AstroFlowNode()
def store_targets(targets: TargetsDict) -> None:
  context.targets = targets
  print(f'### store_targets: Stored {len(targets)} targets; '
        f'total is now {len(context.targets)}')


@AstroFlowNode()
def store_mosaics(mosaics: MosaicsDict) -> None:
  context.mosaics = mosaics
  print(f'### store_mosaics: Stored {len(mosaics)} mosaics; '
        f'total is now {len(context.mosaics)}')


# This graph takes the objects grouped by `build_regroup_graph()` and applies
# user settings (quarantine and subframe selection) on top of them.
#
# TODO: Pass SimpleStacks through `build_stack_user_settings_subgraph()` too.
# Flat-frame stacks could be quarantined, for example.
@AstroFlowNode()
def build_apply_stack_user_settings_subgraph() -> Graph:
  # TODO: This is empty when the preceding tasks don't run.
  stacks = context.grouped_stacks or {}

  stack_ids = [
      s for s in stacks.keys() if isinstance(stacks[s], CompositeStack)
  ]
  stack_id_groups = [
      stack_ids[i::os.cpu_count()] for i in range(os.cpu_count())
  ]
  stack_id_groups = [tuple(group) for group in stack_id_groups]
  print(f'### Grouped {len(stacks)} stacks into {len(stack_id_groups)} groups')
  print(f'### First group {stack_id_groups[0]}')

  graph = Graph({})
  graph[gather_stack_user_settings] = {(build_stack_user_settings_subgraph,
                                        group)
                                       for group in stack_id_groups
                                       if len(group)}
  graph[validate_stacks] = {gather_stack_user_settings}
  graph[store_calibration_stacks] = {validate_stacks}
  graph[quarantine_targets] = {store_calibration_stacks}
  graph[validate_targets] = {quarantine_targets}
  graph[store_targets] = {validate_targets}
  graph[quarantine_mosaics] = {store_targets}
  graph[validate_mosaics] = {quarantine_mosaics}
  graph[store_mosaics] = {validate_mosaics}

  return graph
