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
from targets.validation import validate_targets
from targets.types import TargetHash
from mosaics.compute_graph import build_mosaic_subgraph
from dependency_graph import Graph, TaskDescriptor, make_graph_from_list
from suggestions import make_suggestions
from quarantining import apply_quarantine_to_stack
from quarantining.compute_graph import build_apply_stack_user_settings_subgraph, quarantine_targets, quarantine_mosaics
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
from mosaics.validation import validate_mosaics


@ray.remote
@track_performance
def load_metadata() -> None:
  quarantined_files = context.quarantined_files or set()

  def predicate_fn(image: ImageMetadata) -> bool:
    return not image_types.is_master(
        image.header) and image.content_hash not in quarantined_files

  metadata = get_metadata(
      predicate_fn=predicate_fn,
      post_fn=lambda images: {i.content_hash: i for i in images})

  metadata_size_mb = objsize.get_deep_size(metadata) / 1024 / 1024
  print(f'### load_metadata() size {metadata_size_mb} MB')

  context.set('metadata', metadata, write_mode=WriteMode.MEMORY_ONLY)


@ray.remote
def build_spatial_group_graph(g: tuple[FrameHash]) -> Graph:
  return make_graph_from_list([(identify_stacks, g), validate_stacks,
                               group_stacks_into_targets])


@ray.remote
def gather_spatial_groups() -> None:
  pass


# TODO: Don't call build_spatial_group_graph for spatial groups that are cached
# and unchanged.
@ray.remote
def scatter_spatial_groups() -> Graph:
  graph = Graph({})
  graph[gather_spatial_groups] = {
      (build_spatial_group_graph, g) for g in context.spatial_groups
  }

  return graph


# This graph depends only on the images that are on disk; it doesn't apply any
# user settings or quarantines. Those are applied later as transforms on the
# results produced by this graph.
#
# TODO: Regrouping has to be done whenever there are new files.
@ray.remote
def build_regroup_graph() -> Graph:
  return make_graph_from_list(
      [cluster_by_image_center, scatter_spatial_groups, identify_mosaics])


@ray.remote
def store_targets(ancestor_results: list[Any]) -> None:
  targets = ancestor_results[0]
  context.or_equals('targets', targets, default_value={})
  print(f'### store_targets: Stored {len(targets)} targets; '
        f'total is now {len(context.targets)}')


@ray.remote
def store_mosaics(ancestor_results: list[Any]) -> None:
  mosaics = ancestor_results[0]
  context.or_equals('mosaics', mosaics, default_value={})
  print(f'### store_mosaics: Stored {len(mosaics)} mosaics; '
        f'total is now {len(context.mosaics)}')


# TODO: Move this somewhere more appropriate.
@ray.remote
def apply_settings(ancestor_results: list[CompositeStack]) -> CompositeStack:
  stack = ancestor_results[0]
  if stack is None:
    return None

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

  return replace(stack, contents=new_contents)


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
@ray.remote
def build_stack_user_settings_subgraph(stack_id: StackHash) -> Graph:
  return make_graph_from_list([(apply_quarantine_to_stack, stack_id),
                               select_subframes, associate_with_stack,
                               apply_settings])


@ray.remote
def gather_stack_user_settings(
    ancestor_results: list[Any]) -> CompositeStacksDict:
  stacks = {a.stable_hash: a for a in ancestor_results if a is not None}
  print(f'### gather_stack_user_settings stack ids are {stacks.keys()}')
  context.or_equals('stacks', stacks, default_value={})
  print(f'### gather_stack_user_settings: Stored {len(stacks)} stacks; '
        f'total is now {len(context.stacks)}')

  return stacks


@ray.remote
def build_preview_graph() -> Graph:
  graph = Graph({})
  stack_thumbnail_task = (scatter_stack_thumbnails, QualityLevel.SUBFRAME)
  graph[stack_thumbnail_task] = []

  for mosaic_id in context.mosaics.keys():
    subgraph = ray.get(
        build_mosaic_subgraph.remote(QualityLevel.SUBFRAME, mosaic_id))
    graph |= subgraph

  return graph


@ray.remote
def build_main_processing_graph() -> Graph:
  shopping_cart = context.shopping_cart or []
  print(f'### build_main_processing_graph shopping_cart {shopping_cart}')

  # TODO: Maybe add prioritization options in the UI.
  # TODO: Using ray.get() here is a little awkward.
  graph = Graph({})
  for type, id in shopping_cart:
    if type == 'stack':
      if id not in context.stacks:
        print('### WARNING: stack {id} is in shopping cart but does not exist')
        continue
      subgraph = ray.get(build_stack_subgraph.remote(id))
      graph |= subgraph
    elif type == 'target':
      if id not in context.targets:
        print('### WARNING: target {id} is in shopping cart but does not exist')
        continue
      subgraph = ray.get(build_target_subgraph.remote(TargetHash(id)))
      graph |= subgraph
    if type == 'mosaic':
      if id not in context.mosaics:
        print('### WARNING: mosaic {id} is in shopping cart but does not exist')
        continue
      subgraph = ray.get(
          build_mosaic_subgraph.remote(QualityLevel.MASTER, MosaicHash(id)))
      graph |= subgraph

  return graph


def build_compute_graph(regroup: bool = False,
                        make_previews: bool = True,
                        run_main_processing_graph: bool = True) -> Graph:
  task_list: list[TaskDescriptor] = [(scatter_compute_image_metadata, 'input'),
                                     load_metadata]

  # The regroup graph only looks at light frames only, and ignores calibration
  # frames. All of its findings are written into the given `grouped_*` keys.
  if regroup:
    context.grouped_stacks = {}
    context.grouped_targets = {}
    context.grouped_mosaics = {}
    task_list.append(build_regroup_graph)

  # This looks only at calibration frames.
  # TODO: Don't do this unless there are new calibration frames.
  task_list.append(identify_calibration_stacks)

  # These begin each run empty, and are populated below by the `store_*`
  # functions, after user settings are applied.
  context.stacks = {}
  context.targets = {}
  context.mosaics = {}

  # The tasks that follow consume the raw objects computed from the files
  # on disk, and transform them into the final objects seen by the rest of
  # the graph.
  task_list.append(build_apply_stack_user_settings_subgraph)

  task_list.append(quarantine_targets)
  task_list.append(validate_targets)
  task_list.append(store_targets)

  task_list.append(quarantine_mosaics)
  task_list.append(validate_mosaics)
  task_list.append(store_mosaics)

  task_list.append(make_suggestions)

  if make_previews:
    task_list.append(build_preview_graph)

  if run_main_processing_graph:
    task_list.append(build_main_processing_graph)

  return make_graph_from_list(task_list)


# TODO: Add a cleanup task that deletes unreferenced stuff from the cache.
# Can I use some kind of automatic ref-counting?
