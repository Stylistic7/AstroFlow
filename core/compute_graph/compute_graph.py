import ray

from context import context
from metadata.image_metadata import scatter_compute_image_metadata
from metadata.classes import MetadataDict
from clustering import cluster_by_image_center
from calibration import identify_calibration_stacks
from previews.stack_thumbnails import scatter_stack_thumbnails
from previews.types import QualityLevel
from mosaics.identification import identify_mosaics
from stacks.compute_graph import build_stacking_subgraph
from targets.compute_graph import build_target_subgraph
from targets.types import TargetsDict
from targets.identification import group_stacks_into_targets
from mosaics.compute_graph import build_mosaic_subgraph
from dependency_graph import Graph, TaskDescriptor, make_graph_from_list
from stacks.identification import identify_stacks
from stacks.composite_stack import CompositeStacksDict, SimpleStacksDict
from stacks.validation import validate_stacks
from decorators import cached
from customization.compute_graph import build_customize_graph
from mosaics.types import MosaicsDict
from mosaics.compute_graph import stacks_for_mosaic, targets_for_mosaic
from quarantining.quarantining import quarantine_calibration_stacks
from hashing import hash_list
from metadata.image_metadata import load_metadata


@ray.remote
def store_calibration_stacks(calibration_stacks: SimpleStacksDict) -> None:
  context.calibration_stacks = calibration_stacks
  print(f'### store_calibration_stacks: Stored '
        f'{len(calibration_stacks)} calibration stacks; '
        f'total is now {len(context.calibration_stacks)}')
  return calibration_stacks


@ray.remote
def store_stacks(stacks: CompositeStacksDict) -> None:
  context.grouped_stacks = stacks
  print(f'### store_stacks: Stored '
        f'{len(stacks)} stacks; total is now {len(context.grouped_stacks)}')
  return stacks


@ray.remote
def store_targets(targets: TargetsDict) -> None:
  context.grouped_targets = targets
  print(f'### store_targets: Stored {len(targets)} targets; '
        f'total is now {len(context.grouped_targets)}')
  return targets


@ray.remote
def store_mosaics(mosaics: MosaicsDict) -> None:
  context.grouped_mosaics = mosaics
  print(f'### store_mosaics: Stored {len(mosaics)} mosaics; '
        f'total is now {len(context.grouped_mosaics)}')
  return mosaics


@cached.remote
def gather_spatial_groups(*stacks_in: list[CompositeStacksDict]) -> None:
  stacks = {key: value for d in stacks_in for key, value in d.items()}
  print(f'### gather_spatial_groups returning {len(stacks)} stacks')
  return stacks


# TODO: Don't call build_spatial_group_graph for spatial groups that are cached
# and unchanged.
@cached.remote
def scatter_spatial_groups(spatial_groups) -> Graph:
  graph = Graph({
      gather_spatial_groups: {(identify_stacks, g) for g in spatial_groups},
      validate_stacks: {gather_spatial_groups},
      store_stacks: {validate_stacks}
  })

  return graph


@cached.remote
def regroup_graph_done() -> None:
  pass


# TODO: Move group_stacks_into_targets() back into the parallel path over
# spatial groups.
@cached.remote
def build_regroup_graph(metadata: MetadataDict) -> Graph:
  print(f'build_regroup_graph metadata hash {hash_list([metadata])}')
  return Graph({
      scatter_spatial_groups: {(cluster_by_image_center, metadata)},
      group_stacks_into_targets: {scatter_spatial_groups},
      store_targets: {group_stacks_into_targets},
      identify_mosaics: {store_targets},
      store_mosaics: {identify_mosaics},
      quarantine_calibration_stacks: {identify_calibration_stacks},
      identify_calibration_stacks: {load_metadata},
      store_calibration_stacks: {quarantine_calibration_stacks},
      regroup_graph_done: {store_mosaics, store_calibration_stacks},
  })


@cached.remote
def preview_graph_done() -> None:
  pass


@cached.remote
def build_preview_graph() -> Graph:
  stacks = context.stacks
  targets = context.targets

  tasks = []
  tasks.append((scatter_stack_thumbnails,))

  for mosaic in context.mosaics.values():
    tasks.append(
        (build_mosaic_subgraph, QualityLevel.SUBFRAME, mosaic,
         stacks_for_mosaic(mosaic, stacks,
                           targets), targets_for_mosaic(mosaic, targets)))

  return Graph({preview_graph_done: tasks})


@cached.remote
def main_processing_graph_done() -> None:
  pass


@cached.remote
def build_main_processing_graph() -> Graph:
  stacks = context.stacks
  targets = context.targets
  mosaics = context.mosaics
  shopping_cart = context.shopping_cart or []

  print(f'### build_main_processing_graph shopping_cart {shopping_cart}')

  # TODO: Maybe add prioritization options in the UI.
  graph = Graph({})
  tasks = []
  for type, id in shopping_cart:
    if type == 'stack':
      if id not in stacks:
        print(f'### WARNING: stack {id} is in shopping cart but does not exist')
        continue
      tasks.append((build_stacking_subgraph, stacks[id]))
    elif type == 'target':
      if id not in targets:
        print(
            f'### WARNING: target {id} is in shopping cart but does not exist')
        continue
      tasks.append((build_target_subgraph, targets[id]))
    if type == 'mosaic':
      if id not in mosaics:
        print(
            f'### WARNING: mosaic {id} is in shopping cart but does not exist')
        continue

      mosaic = mosaics[id]
      tasks.append(
          (build_mosaic_subgraph, QualityLevel.MASTER, mosaic,
           stacks_for_mosaic(mosaic, stacks,
                             targets), targets_for_mosaic(mosaic, targets)))

  graph[main_processing_graph_done] = tasks
  return graph


def build_compute_graph() -> Graph:
  task_list: list[TaskDescriptor] = [(scatter_compute_image_metadata, 'input')]

  # The regroup graph only looks at light frames only, and ignores calibration
  # frames. All of its findings are written into the given `grouped_*` keys.
  task_list.append(build_regroup_graph)
  task_list.append(build_customize_graph)
  task_list.append(build_preview_graph)
  task_list.append(build_main_processing_graph)

  graph = make_graph_from_list(task_list)

  return graph


# TODO: Add a cleanup task that deletes unreferenced stuff from the cache.
# Can I use some kind of automatic ref-counting?
