import ray
from dataclasses import replace
from typing import Any

from context import context
from metadata.image_metadata import scatter_compute_image_metadata
from clustering import cluster_by_image_center
from calibration import identify_calibration_stacks
from previews import scatter_stack_thumbnails
from previews.types import QualityLevel
from mosaics.identification import identify_mosaics
from stacks.compute_graph import build_stacking_subgraph
from targets.compute_graph import build_target_subgraph
from targets.identification import group_stacks_into_targets
from mosaics.compute_graph import build_mosaic_subgraph
from dependency_graph import Graph, TaskDescriptor, make_graph_from_list
from quarantining.compute_graph import store_stacks, store_targets, store_mosaics, store_calibration_stacks
from stacks.identification import identify_stacks
import metadata.image_types as image_types
from stacks.composite_stack import CompositeStacksDict
from stacks.validation import validate_stacks
from dependency_graph.astroflow_node import AstroFlowNode
from stacks.customization import build_customize_graph


@AstroFlowNode()
def gather_spatial_groups(*stacks_in: list[CompositeStacksDict]) -> None:
  stacks = {key: value for d in stacks_in for key, value in d.items()}
  print(f'### gather_spatial_groups returning {len(stacks)} stacks')
  return stacks


# TODO: Don't call build_spatial_group_graph for spatial groups that are cached
# and unchanged.
@AstroFlowNode()
def scatter_spatial_groups(spatial_groups) -> Graph:
  graph = Graph({
      gather_spatial_groups: {(identify_stacks, g) for g in spatial_groups},
      validate_stacks: {gather_spatial_groups},
      store_stacks: {validate_stacks}
  })

  return graph


# This graph depends only on the images that are on disk; it doesn't apply any
# user settings or quarantines. Those are applied later as transforms on the
# results produced by this graph.
#
# TODO: Regrouping has to be done whenever there are new files.
@AstroFlowNode()
def build_regroup_graph() -> Graph:
  return make_graph_from_list([
      cluster_by_image_center, scatter_spatial_groups,
      group_stacks_into_targets, store_targets, identify_mosaics, store_mosaics
  ])


@AstroFlowNode()
def preview_graph_done() -> None:
  pass


@AstroFlowNode()
def build_preview_graph() -> Graph:
  tasks = []

  tasks.append((scatter_stack_thumbnails, QualityLevel.SUBFRAME))

  for mosaic in context.mosaics.values():
    tasks.append((build_mosaic_subgraph, QualityLevel.SUBFRAME, mosaic))

  return Graph({preview_graph_done: tasks})


@AstroFlowNode()
def main_processing_graph_done() -> None:
  pass


@AstroFlowNode()
def build_main_processing_graph() -> Graph:
  stacks = context.stacks
  targets = context.targets
  mosaics = context.mosaics
  shopping_cart = context.shopping_cart or []
  print(f'### build_main_processing_graph shopping_cart {shopping_cart}')

  visited_stacks = set()
  visited_targets = set()
  visited_mosaics = set()

  def _visit_stack(stack_hash):
    visited_stacks.add(stack_hash)

  def _visit_target(target_hash):
    visited_targets.add(target_hash)
    for stack in targets[target_hash].stack_ids:
      _visit_stack(stack)

  def _visit_mosaic(mosaic_hash):
    visited_mosaics.add(mosaic_hash)
    for target in mosaics[mosaic_hash].target_ids:
      _visit_target(target)

  # TODO: Maybe add prioritization options in the UI.
  graph = Graph({})
  tasks = []
  for type, id in shopping_cart:
    if type == 'stack':
      if id not in context.stacks:
        print('### WARNING: stack {id} is in shopping cart but does not exist')
        continue
      tasks.append((build_stacking_subgraph, stacks[id]))
      _visit_stack(id)
    elif type == 'target':
      if id not in context.targets:
        print('### WARNING: target {id} is in shopping cart but does not exist')
        continue
      tasks.append((build_target_subgraph, id))
      _visit_target(id)
    if type == 'mosaic':
      if id not in context.mosaics:
        print('### WARNING: mosaic {id} is in shopping cart but does not exist')
        continue
      tasks.append((build_mosaic_subgraph, QualityLevel.MASTER, mosaics[id]))
      _visit_mosaic(id)

  graph[main_processing_graph_done] = tasks
  return graph


def build_compute_graph(regroup: bool = False,
                        make_previews: bool = True,
                        run_main_processing_graph: bool = True) -> Graph:
  task_list: list[TaskDescriptor] = [(scatter_compute_image_metadata, 'input')]

  # The regroup graph only looks at light frames only, and ignores calibration
  # frames. All of its findings are written into the given `grouped_*` keys.
  if regroup:
    # TODO: The @AstroFlowNode decorator works with values passed down through
    # the dependency graph as return values and arguments. These context-based
    # values are basically "side effects." When the decorator doesn't run
    # the functions, these side effects don't happen.
    #
    context.calibration_stacks = {}
    context.grouped_stacks = {}
    context.grouped_targets = {}
    context.grouped_mosaics = {}
    task_list.append(build_regroup_graph)

  # TODO: Don't do this unless there are new calibration frames.
  task_list.append(identify_calibration_stacks)
  task_list.append(store_calibration_stacks)

  task_list.append(build_customize_graph)

  if make_previews:
    task_list.append(build_preview_graph)

  if run_main_processing_graph:
    task_list.append(build_main_processing_graph)

  graph = make_graph_from_list(task_list)

  return graph


# TODO: Add a cleanup task that deletes unreferenced stuff from the cache.
# Can I use some kind of automatic ref-counting?
