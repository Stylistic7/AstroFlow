import ray

from context import context
from metadata.image_metadata import scatter_compute_image_metadata
from clustering import cluster_by_image_center
from calibration import identify_calibration_stacks, associate_with_stacks
from previews import scatter_stack_thumbnails
from mosaics.identification import identify_mosaics
from stacks.compute_graph import build_stack_subgraph, scatter_identify_stacks
from targets.compute_graph import build_target_subgraph
from mosaics.compute_graph import build_mosaic_subgraph
from dependency_graph import Graph, make_graph_from_list, print_graph
from suggestions import make_suggestions
from quarantining import quarantine_stacks, quarantine_targets, quarantine_mosaics


@ray.remote
def maybe_build_regroup_graph():
  if context.regroup:
    return make_graph_from_list([
        cluster_by_image_center, scatter_identify_stacks,
        identify_calibration_stacks, associate_with_stacks, identify_mosaics
    ])
  else:
    return {}


@ray.remote
def quarantine():
  return make_graph_from_list(
      [quarantine_stacks, quarantine_targets, quarantine_mosaics])


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


def build_compute_graph():
  return make_graph_from_list([
      (scatter_compute_image_metadata, 'input'),
      maybe_build_regroup_graph,
      quarantine,
      make_suggestions,
      build_preview_graph,
      # This is the heavyweight graph that runs PixInsight and may take days.
      build_main_processing_graph,
  ])


# TODO: Add a cleanup task that deletes unreferenced stuff from the cache.
# Can I use some kind of automatic ref-counting?
