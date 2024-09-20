import ray
import objsize

from context import context, WriteMode
from metadata.image_metadata import scatter_compute_image_metadata
from clustering import cluster_by_image_center
from calibration import identify_calibration_stacks, associate_with_stacks
from previews import scatter_stack_thumbnails
from mosaics.identification import identify_mosaics
from stacks.compute_graph import build_stack_subgraph
from targets.compute_graph import build_target_subgraph
from targets.identification import group_stacks_into_targets
from mosaics.compute_graph import build_mosaic_subgraph
from dependency_graph import Graph, make_graph_from_list, print_graph
from suggestions import make_suggestions
from quarantining import quarantine_stacks
from quarantining.compute_graph import delete_quarantined_targets, delete_quarantined_mosaics
from quarantining.subframe_selection import select_subframes
from stacks.identification import identify_stacks
import metadata.image_types as image_types
from metadata.actor import get_metadata
from stacks import update_state
from profiling import track_performance


@ray.remote
@track_performance
def load_metadata():
  quarantined_files = context.quarantined_files or set()

  def predicate_fn(image):
    return not image_types.is_master(
        image.header) and image.content_hash not in quarantined_files

  metadata = get_metadata(
      predicate_fn=predicate_fn,
      post_fn=lambda images: {i.content_hash: i for i in images})

  metadata_size_mb = objsize.get_deep_size(metadata) / 1024 / 1024
  print(f'### load_metadata() size {metadata_size_mb} MB')

  context.set('metadata', metadata, write_mode=WriteMode.MEMORY_ONLY)


@ray.remote
def build_spatial_group_graph(g):
  name, frames = g
  print(f'### build_spatial_group_graph running for '
        f'spatial group {name} of length {len(frames)}')
  return make_graph_from_list([(identify_stacks, g), associate_with_stacks,
                               quarantine_stacks, select_subframes,
                               update_state, group_stacks_into_targets,
                               delete_quarantined_targets])


@ray.remote
def gather_spatial_groups():
  pass


# TODO: Don't call build_spatial_group_graph for spatial groups that are cached
# and unchanged.
@ray.remote
def scatter_spatial_groups():
  graph = Graph()
  graph[gather_spatial_groups] = {
      (build_spatial_group_graph, g) for g in context.spatial_groups
  }

  return graph


# TODO: Regrouping has to be done whenever there are new files, or whenever
# quarantining or (global) subframe selection options change.
@ray.remote
def build_regroup_graph():
  return make_graph_from_list([
      identify_calibration_stacks, cluster_by_image_center,
      scatter_spatial_groups, identify_mosaics, delete_quarantined_mosaics
  ])


@ray.remote
def build_preview_graph():
  graph = Graph()
  stack_thumbnail_task = (scatter_stack_thumbnails, 'subframe')
  graph[stack_thumbnail_task] = []

  for mosaic_id in context.mosaics.keys():
    graph |= ray.get(build_mosaic_subgraph.remote('subframe', mosaic_id))

  print('### Finished build_preview_graph, graph is:')
  print_graph(graph)

  return graph


@ray.remote
def build_main_processing_graph():
  shopping_cart = context.shopping_cart or []
  print(f'### build_main_processing_graph shopping_cart {shopping_cart}')

  # TODO: Maybe add prioritization options in the UI.
  # TODO: Using ray.get() here is a little awkward.
  graph = Graph()
  for type, id in shopping_cart:
    if type == 'stack':
      if id not in context.stacks:
        print('### WARNING: stack {id} is in shopping cart but does not exist')
        continue
      graph |= ray.get(build_stack_subgraph.remote(id))
    elif type == 'target':
      if id not in context.targets:
        print('### WARNING: target {id} is in shopping cart but does not exist')
        continue
      graph |= ray.get(build_target_subgraph.remote(id))
    if type == 'mosaic':
      if id not in context.mosaics:
        print('### WARNING: mosaic {id} is in shopping cart but does not exist')
        continue
      graph |= ray.get(build_mosaic_subgraph.remote('master', id))

  print(f'### Finished composing main processing graph')
  print_graph(graph)

  return graph


def build_compute_graph(regroup=False,
                        make_previews=True,
                        run_main_processing_graph=True):
  task_list = [(scatter_compute_image_metadata, 'input'), load_metadata]

  if regroup:
    context.stacks = {}
    context.targets = {}
    context.mosaics = {}
    task_list.append(build_regroup_graph)

  task_list.append(make_suggestions)

  if make_previews:
    task_list.append(build_preview_graph)

  if run_main_processing_graph:
    task_list.append(build_main_processing_graph)

  return make_graph_from_list(task_list)


# TODO: Add a cleanup task that deletes unreferenced stuff from the cache.
# Can I use some kind of automatic ref-counting?
