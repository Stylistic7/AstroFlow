from pathlib import Path
import ray
import numpy as np
from spherical_geometry.polygon import SingleSphericalPolygon

from context import context
from stacks.stacking import stack_files
from stacks.composite_stack import CompositeStack
from stacks.post_process import post_process_stack
from stacks.convert_to_dict import convert_to_dict
from dependency_graph import Graph, Node
from decorators import cached
from metadata.image_metadata import compute_image_metadata
from customization.types import CustomizedStack
from customization.subframe_selection import get_light_frames_sorted_by_stars
from previews.types import QualityLevel
from dependency_graph import run_graph


@ray.remote
def calibration_done(stack: CustomizedStack) -> None:
  pass


@ray.remote
def build_calibration_stack_subgraph(stack: CustomizedStack) -> Graph:
  calibration_stacks = context.calibration_stacks
  return Graph({
      (calibration_done, stack): {(stack_files,
                                   calibration_stacks[d['stack_id']])
                                  for substack in stack.contents.values()
                                  for d in substack.dependencies.values()
                                  if 'stack_id' in d}
  })


@ray.remote
def stack_runs_done(stack: CustomizedStack) -> None:
  pass


# This function accepts a fully customized stack, after settings and calibration
# frames have been associated.
@cached.remote
def build_stacking_subgraph(stack: CustomizedStack) -> Graph:
  graph = Graph({})

  calibration_subgraph = (build_calibration_stack_subgraph, stack)

  stack_task = (stack_files, stack)
  stacking_metadata_task = Node(compute_image_metadata,
                                name=f'stacking_{stack.stable_hash}')
  post_process_task = Node(post_process_stack, stack)
  post_process_metadata_task = Node(compute_image_metadata,
                                    name=f'post_process_{stack.stable_hash}')

  graph[stack_task] = {calibration_subgraph}
  graph[stacking_metadata_task] = {stack_task}
  graph[post_process_task] = {stacking_metadata_task}
  graph[post_process_metadata_task] = {post_process_task}

  return graph


# TODO: Calibrate the subframe returned here.
@cached.remote
def get_stack_subframe_image(stack: CompositeStack) -> Path:
  assert stack is not None
  light_frames_by_stars = get_light_frames_sorted_by_stars(stack)
  best_light_frame = stack.all_metadata()[light_frames_by_stars[0]]
  return Path(best_light_frame.filename)


@cached.remote
def get_stack_image(mode: QualityLevel, stack: CompositeStack) -> Graph:
  assert stack is not None
  if mode == QualityLevel.SUBFRAME:
    return Graph({(get_stack_subframe_image, stack): {}})
  else:
    return Graph({
        (post_process_stack, stack): {(build_stacking_subgraph, stack)}
    })


@ray.remote
def process_stacks_for_ui() -> None:

  def my_transform(v):
    if type(v) is np.ndarray:
      return v.tolist()
    if type(v) is SingleSphericalPolygon:
      return v.points.tolist()
    return v

  def prepare_stack(stack: CustomizedStack):
    result = convert_to_dict(stack, [['contents', '*', 'metadata']],
                             transform=my_transform)

    filenames = []
    for s in stack.contents.values():
      filenames += [m.filename for m in s.metadata.values()]

    result['filenames'] = filenames

    graph = ray.get(build_stacking_subgraph.remote(stack))

    graph_results, _ = run_graph(graph, dry_run=True)
    result['contains_exception'] = graph_results.contains_exception()
    result['is_complete'] = graph_results.is_complete()

    return result

  # TODO: Parallelize this.
  context.ui_stacks = {h: prepare_stack(s) for h, s in context.stacks.items()}
