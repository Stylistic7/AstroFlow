import ray

from context import context
from metadata.image_metadata import scatter_compute_image_metadata
from clustering import cluster_by_image_center
from calibration import identify_calibration_stacks
from previews.stack_thumbnails import scatter_stack_thumbnails
from previews.types import QualityLevel
from mosaics.identification import identify_mosaics
from stacks.compute_graph import build_stacking_subgraph, process_stacks_for_ui
from targets.compute_graph import build_target_subgraph, process_targets_for_ui
from targets.identification import group_stacks_into_targets
from targets.types import TargetsDict
from mosaics.compute_graph import build_mosaic_subgraph, process_mosaics_for_ui
from mosaics.types import MosaicsDict
from dependency_graph import Graph, chain
from stacks.identification import identify_stacks
from stacks.composite_stack import CompositeStacksDict, SimpleStacksDict
from stacks.validation import validate_stacks
from decorators import cached
from customization.compute_graph import scatter_customize_stacks, scatter_customize_targets, scatter_customize_mosaics
from mosaics.compute_graph import stacks_for_mosaic, targets_for_mosaic
from metadata.image_metadata import load_metadata
from hashing.types import LargeObject
from quarantining.compute_graph import quarantine_stacks, quarantine_calibration_stacks, quarantine_targets, quarantine_mosaics
from customization.settings import Settings
from customization.types import CustomizedStacksDict


@ray.remote
def store_calibration_stacks(calibration_stacks: SimpleStacksDict) -> None:
  context.calibration_stacks = calibration_stacks
  print(f'### store_calibration_stacks: Stored '
        f'{len(calibration_stacks)} calibration stacks')
  return calibration_stacks


@ray.remote
def store_stacks(stacks: CustomizedStacksDict):
  context.stacks = stacks
  print(f'### store_stacks: Stored {len(stacks)} stacks')
  return stacks


@ray.remote
def store_targets(targets: TargetsDict):
  context.targets = targets
  print(f'### store_targets: Stored {len(targets)} targets')
  return targets


@ray.remote
def store_mosaics(mosaics: MosaicsDict):
  context.mosaics = mosaics
  print(f'### store_mosaics: Stored {len(mosaics)} mosaics')
  return mosaics


@cached.remote
def gather_spatial_groups(
    *stacks_in: list[CompositeStacksDict]) -> CompositeStacksDict:
  return {key: value for d in stacks_in for key, value in d.items()}


@ray.remote
def scatter_spatial_groups(metadata_ref: LargeObject, spatial_groups) -> Graph:
  return Graph({
      gather_spatial_groups: [
          (identify_stacks, metadata_ref, g) for g in spatial_groups
      ],
      validate_stacks: {gather_spatial_groups},
  })


# TODO: Add step that cleans metadata, removing any images that won't make it
# through the remaining steps, and alerting the user.
@ray.remote
def build_regroup_graph(metadata_ref: LargeObject) -> Graph:
  quarantined_stacks = context.quarantined_stacks or tuple()
  quarantined_calibration_stacks = context.quarantined_calibration_stacks or tuple(
  )
  quarantined_targets = context.quarantined_targets or tuple()
  quarantined_mosaics = context.quarantined_mosaics or tuple()
  settings = context.settings or Settings()

  scatter_spatial_groups_task = (scatter_spatial_groups, metadata_ref)
  cluster_by_image_center_task = (cluster_by_image_center, metadata_ref)
  identify_calibration_stacks_task = (identify_calibration_stacks, metadata_ref)
  group_stacks_into_targets_task = (group_stacks_into_targets, metadata_ref)

  quarantine_calibration_stacks_task = (quarantine_calibration_stacks,
                                        quarantined_calibration_stacks)
  quarantine_stacks_task = (quarantine_stacks, quarantined_stacks)
  quarantine_targets_task = (quarantine_targets, quarantined_stacks,
                             quarantined_targets)
  quarantine_mosaics_task = (quarantine_mosaics, quarantined_targets,
                             quarantined_mosaics)

  scatter_customize_stacks_task = (scatter_customize_stacks, settings)
  scatter_customize_targets_task = (scatter_customize_targets, settings)
  scatter_customize_mosaics_task = (scatter_customize_mosaics, settings)

  return Graph({
      scatter_spatial_groups_task: {cluster_by_image_center_task},
      quarantine_calibration_stacks_task: {identify_calibration_stacks_task},
      store_calibration_stacks: {quarantine_calibration_stacks_task},
      process_stacks_for_ui: {scatter_spatial_groups_task},
      quarantine_stacks_task: {scatter_spatial_groups_task},
      scatter_customize_stacks_task: [
          quarantine_stacks_task, quarantine_calibration_stacks_task
      ],
      store_stacks: {scatter_customize_stacks_task},
      group_stacks_into_targets_task: {scatter_customize_stacks_task},
      scatter_customize_targets_task: {group_stacks_into_targets_task},
      process_targets_for_ui: {group_stacks_into_targets_task},
      quarantine_targets_task: [
          scatter_spatial_groups_task, scatter_customize_targets_task
      ],
      store_targets: {quarantine_targets_task},
      identify_mosaics: {quarantine_targets_task},
      scatter_customize_mosaics_task: {identify_mosaics},
      quarantine_mosaics_task: [
          scatter_customize_targets_task, scatter_customize_mosaics_task
      ],
      store_mosaics: {quarantine_mosaics_task},
      process_mosaics_for_ui: [
          scatter_spatial_groups_task, group_stacks_into_targets_task,
          identify_mosaics
      ]
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
  mode = context.forced_mode or QualityLevel.MASTER

  print(f'### build_main_processing_graph shopping_cart {shopping_cart}')

  # shopping_cart += [('stack', stack_id) for stack_id in stacks.keys()]
  # shopping_cart += [('target', target_id) for target_id in targets.keys()]
  # shopping_cart += [('mosaic', mosaic_id) for mosaic_id in mosaics.keys()]

  # shopping_cart += [('mosaic', '75a1da6463e118e6')]
  # shopping_cart += [('mosaic', '0ee57ba3ed4695a3')]

  # TODO: Maybe add prioritization options in the UI.
  graph = Graph({})
  tasks = []
  for type, id in shopping_cart:
    if type == 'stack':
      if id not in stacks:
        print(f'### WARNING: stack {id} is in shopping cart but does not exist')
        continue
      tasks.append((build_stacking_subgraph, mode, stacks[id]))
    elif type == 'target':
      if id not in targets:
        print(
            f'### WARNING: target {id} is in shopping cart but does not exist')
        continue
      tasks.append((build_target_subgraph, mode, targets[id]))
    if type == 'mosaic':
      if id not in mosaics:
        print(
            f'### WARNING: mosaic {id} is in shopping cart but does not exist')
        continue

      mosaic = mosaics[id]
      tasks.append(
          (build_mosaic_subgraph, mode, mosaic,
           stacks_for_mosaic(mosaic, stacks,
                             targets), targets_for_mosaic(mosaic, targets)))

  graph[main_processing_graph_done] = tasks

  return graph


def build_compute_graph() -> Graph:
  return Graph({
      **chain(
          scatter_compute_image_metadata,
          build_regroup_graph,
          build_preview_graph,
          build_main_processing_graph,
      )
  })


# TODO: Add a cleanup task that deletes unreferenced stuff from the cache.
# Can I use some kind of automatic ref-counting?
