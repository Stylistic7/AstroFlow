from pathlib import Path

from context import context
from stacks.stacking import stack_files
from stacks.composite_stack import CompositeStack
from stacks.post_process import post_process_stack
from dependency_graph import Graph, Node
from decorators.task_decorator import task
from metadata.image_metadata import compute_image_metadata, load_metadata
from customization.types import CustomizedStack
from customization.subframe_selection import get_light_frames_sorted_by_stars
from previews import QualityLevel


# This function accepts a fully customized stack, after settings and calibration
# frames have been associated.
@task.remote
def build_stacking_subgraph(stack: CustomizedStack) -> Graph:
  calibration_stacks = context.calibration_stacks

  graph = Graph({})

  stack_task = (stack_files, stack)
  stacking_metadata_task = Node(compute_image_metadata,
                                name=f'stacking_{stack.stable_hash}')
  post_process_task = Node(post_process_stack, stack)
  post_process_metadata_task = Node(compute_image_metadata,
                                    name=f'post_process_{stack.stable_hash}')

  graph[stack_task] = {(stack_files, calibration_stacks[d])
                       for substack in stack.contents.values()
                       for d in substack.dependencies.values()}

  graph[stacking_metadata_task] = {stack_task}
  graph[post_process_task] = {stacking_metadata_task}
  graph[post_process_metadata_task] = {post_process_task}
  graph[load_metadata] = {post_process_metadata_task}

  return graph


# TODO: Calibrate the subframe returned here.s
@task.remote
def get_stack_subframe_image(stack: CompositeStack) -> Path:
  assert stack is not None
  light_frames_by_stars = get_light_frames_sorted_by_stars(stack)
  best_light_frame = stack.all_metadata()[light_frames_by_stars[0]]
  print(f'### get_stack_subframe_image returning {best_light_frame.filename}')
  return best_light_frame.filename


@task.remote
def get_stack_image(mode: QualityLevel, stack: CompositeStack) -> Graph:
  assert stack is not None
  if mode == QualityLevel.SUBFRAME:
    return Graph({(get_stack_subframe_image, stack): {}})
  else:
    return Graph({
        (post_process_stack, stack): {(build_stacking_subgraph, stack)}
    })
