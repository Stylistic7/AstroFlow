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
from quarantining.compute_graph import quarantine_targets, quarantine_mosaics
from quarantining.subframe_selection import select_subframes
from stacks.identification import identify_stacks
import metadata.image_types as image_types
from metadata.actor import get_metadata
from stacks import update_state, validate_stacks
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
  return make_graph_from_list([(identify_stacks, g), group_stacks_into_targets])


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


# This graph depends only on the images that are on disk; it doesn't apply any
# user settings or quarantines. Those are applied later as transforms on the
# results produced by this graph.
#
# TODO: Regrouping has to be done whenever there are new files.
@ray.remote
def build_regroup_graph():
  return make_graph_from_list(
      [cluster_by_image_center, scatter_spatial_groups, identify_mosaics])


@ray.remote
def store_stacks(ancestor_results):
  stacks = ancestor_results[0]
  context.or_equals('stacks', stacks, default_value={})
  print(f'### store_stacks: Stored {len(stacks)} stacks; '
        f'total is now {len(context.stacks)}')


@ray.remote
def store_targets(ancestor_results):
  targets = ancestor_results[0]
  context.or_equals('targets', targets, default_value={})
  print(f'### store_targets: Stored {len(targets)} targets; '
        f'total is now {len(context.targets)}')


@ray.remote
def store_mosaics(ancestor_results):
  mosaics = ancestor_results[0]
  context.or_equals('mosaics', mosaics, default_value={})
  print(f'### store_mosaics: Stored {len(mosaics)} mosaics; '
        f'total is now {len(context.mosaics)}')


# This graph applies user settings to a single stack in the `grouped_stacks`
# dict, passing it through graph arguments. If the stack is quarantined, or
# becomes empty after subframe selection, it will not make it into the final
# `stacks` dict.
@ray.remote
def build_stack_user_settings_subgraph(stack_id):
  return make_graph_from_list([(quarantine_stacks, stack_id), select_subframes,
                               associate_with_stacks, store_stacks])


@ray.remote
def gather_stack_user_settings():
  pass


# This graph takes the objects grouped by `build_regroup_graph()` and applies
# user settings (quarantine and subframe selection) on top of them.
@ray.remote
def build_apply_stack_user_settings_subgraph():
  graph = Graph()
  graph[gather_stack_user_settings] = {
      (build_stack_user_settings_subgraph, stack_id)
      for stack_id in context.grouped_stacks.keys()
  }

  return graph


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
  task_list.append(validate_stacks)

  task_list.append(quarantine_targets)
  task_list.append(store_targets)

  task_list.append(quarantine_mosaics)
  task_list.append(store_mosaics)

  task_list.append(make_suggestions)

  if make_previews:
    task_list.append(build_preview_graph)

  if run_main_processing_graph:
    task_list.append(build_main_processing_graph)

  return make_graph_from_list(task_list)


# TODO: Add a cleanup task that deletes unreferenced stuff from the cache.
# Can I use some kind of automatic ref-counting?
