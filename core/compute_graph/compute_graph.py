import ray
import objsize  # type: ignore
from dataclasses import replace
from typing import Any

from context import context, WriteMode
from metadata.image_metadata import scatter_compute_image_metadata
from clustering import cluster_by_image_center
from calibration import identify_calibration_stacks, associate_with_stack
from previews import scatter_stack_thumbnails
from previews.types import QualityLevel
from mosaics.identification import identify_mosaics
from stacks.compute_graph import build_stack_subgraph
from targets.compute_graph import build_target_subgraph
from targets.identification import group_stacks_into_targets
from targets.types import TargetHash
from mosaics.compute_graph import build_mosaic_subgraph
from dependency_graph import Graph, TaskDescriptor, make_graph_from_list
from suggestions import make_suggestions
from quarantining import apply_quarantine_to_stack
from quarantining.compute_graph import store_stacks, store_targets, store_mosaics, build_apply_stack_user_settings_subgraph, quarantine_targets, quarantine_mosaics, store_calibration_stacks
from quarantining.subframe_selection import select_subframes
from stacks.identification import identify_stacks
import metadata.image_types as image_types
from metadata.classes import ImageMetadata, FrameHash
from metadata.actor import get_metadata
from stacks.composite_stack import CompositeStack, CompositeStacksDict
from stacks.types import StackHash
from stacks.validation import validate_stacks
from profiling import track_performance
from mosaics.types import MosaicHash
from dependency_graph.astroflow_node import AstroFlowNode
from targets import Target


@AstroFlowNode(cache_result=False)
def load_metadata() -> None:
  quarantined_files = context.quarantined_files or set()

  def predicate_fn(image: ImageMetadata) -> bool:
    return image.content_hash not in quarantined_files

  metadata = get_metadata(
      predicate_fn=predicate_fn,
      post_fn=lambda images: {i.content_hash: i for i in images})

  metadata_size_mb = objsize.get_deep_size(metadata) / 1024 / 1024
  print(f'### load_metadata() size {metadata_size_mb} MB')

  context.set('metadata', metadata, write_mode=WriteMode.MEMORY_ONLY)


@AstroFlowNode()
def build_spatial_group_graph(g: tuple[FrameHash]) -> Graph:
  return make_graph_from_list([(identify_stacks, g), validate_stacks,
                               store_stacks, group_stacks_into_targets])


@AstroFlowNode()
def gather_spatial_groups(*targets_in: list[Target]) -> None:
  print(f'### gather_spatial_groups targets_in {targets_in}')
  targets = {key: value for d in targets_in for key, value in d.items()}
  print(f'### gather_spatial_groups target ids are {targets.keys()}')
  return targets


# TODO: Don't call build_spatial_group_graph for spatial groups that are cached
# and unchanged.
@AstroFlowNode()
def scatter_spatial_groups(spatial_groups) -> Graph:
  graph = Graph({})
  graph[gather_spatial_groups] = {
      (build_spatial_group_graph, g) for g in spatial_groups
  }

  return graph


# This graph depends only on the images that are on disk; it doesn't apply any
# user settings or quarantines. Those are applied later as transforms on the
# results produced by this graph.
#
# TODO: Regrouping has to be done whenever there are new files.
@AstroFlowNode()
def build_regroup_graph() -> Graph:
  return make_graph_from_list([
      cluster_by_image_center, scatter_spatial_groups, store_targets,
      identify_mosaics, store_mosaics
  ])


@AstroFlowNode()
def preview_graph_done() -> None:
  pass


@AstroFlowNode()
def build_preview_graph() -> Graph:
  graph = Graph({})
  stack_thumbnail_task = (scatter_stack_thumbnails, QualityLevel.SUBFRAME)
  graph[stack_thumbnail_task] = []

  subgraphs = []
  for mosaic in context.mosaics.values():
    subgraphs.append((build_mosaic_subgraph, QualityLevel.SUBFRAME, mosaic))
  graph[preview_graph_done] = subgraphs

  return graph


@AstroFlowNode()
def main_processing_graph_done() -> None:
  pass


@AstroFlowNode()
def build_main_processing_graph() -> Graph:
  mosaics = context.mosaics
  shopping_cart = context.shopping_cart or []
  print(f'### build_main_processing_graph shopping_cart {shopping_cart}')

  # TODO: Maybe add prioritization options in the UI.
  graph = Graph({})
  tasks = []
  for type, id in shopping_cart:
    if type == 'stack':
      if id not in context.stacks:
        print('### WARNING: stack {id} is in shopping cart but does not exist')
        continue
      tasks.append((build_stack_subgraph, id))
    elif type == 'target':
      if id not in context.targets:
        print('### WARNING: target {id} is in shopping cart but does not exist')
        continue
      tasks.append((build_target_subgraph, id))
    if type == 'mosaic':
      if id not in context.mosaics:
        print('### WARNING: mosaic {id} is in shopping cart but does not exist')
        continue
      tasks.append((build_mosaic_subgraph, QualityLevel.MASTER, mosaics[id]))

  graph[main_processing_graph_done] = tasks
  return graph


def build_compute_graph(regroup: bool = False,
                        make_previews: bool = True,
                        run_main_processing_graph: bool = True) -> Graph:
  task_list: list[TaskDescriptor] = [(scatter_compute_image_metadata, 'input'),
                                     load_metadata]

  # The regroup graph only looks at light frames only, and ignores calibration
  # frames. All of its findings are written into the given `grouped_*` keys.
  if regroup:
    # TODO: The @AstroFlowNode decorator works with values passed down through
    # the dependency graph as return values and arguments. These context-based
    # values are basically "side effects." When the decorator doesn't run
    # the functions, these side effects don't happen.
    #
    context.calibration_stacks = {}
    context.stacks = {}
    context.targets = {}
    context.mosaics = {}
    task_list.append(build_regroup_graph)

  # This looks only at calibration frames.
  # TODO: Don't do this unless there are new calibration frames.
  task_list.append(identify_calibration_stacks)
  task_list.append(store_calibration_stacks)

  # The tasks that follow consume the raw objects computed from the files
  # on disk, and transform them into the final objects seen by the rest of
  # the graph.
  task_list.append(build_apply_stack_user_settings_subgraph)

  task_list.append(make_suggestions)

  if make_previews:
    task_list.append(build_preview_graph)

  if run_main_processing_graph:
    task_list.append(build_main_processing_graph)

  return make_graph_from_list(task_list)


# TODO: Add a cleanup task that deletes unreferenced stuff from the cache.
# Can I use some kind of automatic ref-counting?
