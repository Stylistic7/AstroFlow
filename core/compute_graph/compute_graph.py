import ray

from context import context
from metadata.image_metadata import scatter_compute_image_metadata
from metadata.classes import MetadataDict
from clustering import cluster_by_image_center
from calibration import identify_calibration_stacks
from previews.stack_thumbnails import scatter_stack_thumbnails
from previews.types import QualityLevel
from mosaics.identification import identify_mosaics
from stacks.compute_graph import build_stacking_subgraph, process_stacks_for_ui
from targets.compute_graph import build_target_subgraph, process_targets_for_ui
from targets.types import TargetsDict
from targets.identification import group_stacks_into_targets
from mosaics.compute_graph import build_mosaic_subgraph, process_mosaics_for_ui
from dependency_graph import Graph, chain
from stacks.identification import identify_stacks
from stacks.composite_stack import CompositeStacksDict, SimpleStacksDict
from stacks.validation import validate_stacks
from decorators import cached
from customization.compute_graph import build_customize_graph
from mosaics.types import MosaicsDict
from mosaics.compute_graph import stacks_for_mosaic, targets_for_mosaic
from quarantining.quarantining import quarantine_calibration_stacks
from metadata.image_metadata import load_metadata
from hashing.types import LargeObject


@ray.remote
def store_calibration_stacks(calibration_stacks: SimpleStacksDict) -> None:
  context.calibration_stacks = calibration_stacks
  print(f'### store_calibration_stacks: Stored '
        f'{len(calibration_stacks)} calibration stacks')
  return calibration_stacks


@ray.remote
def store_stacks(stacks: CompositeStacksDict) -> None:
  context.grouped_stacks = stacks
  print(f'### store_stacks: Stored {len(stacks)} stacks')
  return stacks


@ray.remote
def store_targets(targets: TargetsDict) -> None:
  context.grouped_targets = targets
  print(f'### store_targets: Stored {len(targets)} targets')
  return targets


@ray.remote
def store_mosaics(mosaics: MosaicsDict) -> None:
  context.grouped_mosaics = mosaics
  print(f'### store_mosaics: Stored {len(mosaics)} mosaics')
  return mosaics


@cached.remote
def gather_spatial_groups(*stacks_in: list[CompositeStacksDict]) -> None:
  stacks = {key: value for d in stacks_in for key, value in d.items()}
  return stacks


# TODO: Don't call build_spatial_group_graph for spatial groups that are cached
# and unchanged.
@cached.remote
def scatter_spatial_groups(metadata_ref: LargeObject, spatial_groups) -> Graph:
  graph = Graph({
      gather_spatial_groups: {
          (identify_stacks, metadata_ref, g) for g in spatial_groups
      },
      validate_stacks: {gather_spatial_groups},
      store_stacks: {validate_stacks}
  })

  return graph


@cached.remote
def regroup_graph_done() -> None:
  pass


# TODO: Add step that cleans metadata, removing any images that won't make it
# through the remaining steps, and alerting the user.
@cached.remote
def build_regroup_graph(metadata_ref: LargeObject) -> Graph:
  return Graph({
      **chain(
          (cluster_by_image_center, metadata_ref),
          (scatter_spatial_groups, metadata_ref),
          (group_stacks_into_targets, metadata_ref),
          store_targets,
          identify_mosaics,
          store_mosaics,
      ),
      **chain(
          (identify_calibration_stacks, metadata_ref),
          quarantine_calibration_stacks,
          store_calibration_stacks,
      ),
      regroup_graph_done: {store_mosaics, store_calibration_stacks},
  })


@cached.remote
def preview_graph_done() -> None:
  pass


@ray.remote
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


@ray.remote
def main_processing_graph_done() -> None:
  pass


@ray.remote
def build_main_processing_graph() -> Graph:
  stacks = context.stacks
  targets = context.targets
  mosaics = context.mosaics
  shopping_cart = context.shopping_cart or []

  print(f'### build_main_processing_graph shopping_cart {shopping_cart}')

  # shopping_cart += [('stack', stack_id) for stack_id in stacks.keys()]
  # shopping_cart += [('target', target_id) for target_id in targets.keys()]
  # shopping_cart += [('mosaic', mosaic_id) for mosaic_id in mosaics.keys()]

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

  metadata_task = (scatter_compute_image_metadata, 'all')

  graph[main_processing_graph_done] = tasks
  graph[metadata_task] = {main_processing_graph_done}
  graph[load_metadata] = {metadata_task}

  return graph


def build_compute_graph() -> Graph:
  return Graph({
      **chain(
          (scatter_compute_image_metadata, 'all'),
          build_regroup_graph,
          build_customize_graph,
          process_stacks_for_ui,
          process_targets_for_ui,
          process_mosaics_for_ui,
          build_preview_graph,
          build_main_processing_graph,
      )
  })


# TODO: Add a cleanup task that deletes unreferenced stuff from the cache.
# Can I use some kind of automatic ref-counting?
