import ray
from pathlib import Path

from context import context
from stacks.types import StackHash
from stacks.stacking import stack_files
from stacks.composite_stack import CompositeStack, SimpleStacksDict
from stacks.post_process import post_process_stack
from dependency_graph import Graph, Node, make_graph_from_list
from dependency_graph.astroflow_node import AstroFlowNode
from metadata.image_metadata import compute_image_metadata, load_metadata
from quarantining.settings import Settings, StackSettings
from calibration import identify_calibration_stacks, associate_with_stack
from quarantining.compute_graph import store_calibration_stacks, apply_settings
from previews.stack_thumbnails import create_stack_subframe_thumbnails
from stacks.customization import CustomizedStack, customize_stack
from previews import QualityLevel
from quarantining.subframe_selection import get_light_frames_sorted_by_stars


# This function accepts a fully customized stack, after settings and calibration
# frames have been associated.
@AstroFlowNode(cache_result=True)
def build_stacking_subgraph(stack: CustomizedStack) -> Graph:
  stacks = context.stacks

  graph = Graph({})

  stack_task = (stack_files, stack)
  stacking_metadata_task = Node(compute_image_metadata,
                                name=f'stacking_{stack.stable_hash}')
  post_process_metadata_task = Node(compute_image_metadata,
                                    name=f'post_process_{stack.stable_hash}')

  # TODO: Bug fix: Settings are not getting applied to substacks.
  graph[stack_task] = {(stack_files, stacks[d])
                       for substack in stack.contents.values()
                       for d in substack.dependencies.values()
                       if stacks[d].image_type != 'light'}

  graph[stacking_metadata_task] = {stack_task}
  graph[post_process_stack] = {stacking_metadata_task}
  graph[post_process_metadata_task] = {post_process_stack}
  graph[load_metadata] = {post_process_metadata_task}

  return graph


@AstroFlowNode()
def get_stack_subframe_image(stack: CompositeStack) -> Path:
  light_frames_by_stars = get_light_frames_sorted_by_stars(stack.all_frames())
  best_light_frame = light_frames_by_stars[0]
  return best_light_frame.filename


@AstroFlowNode()
def get_stack_image(mode: QualityLevel, stack: CompositeStack) -> Graph:
  if mode == QualityLevel.SUBFRAME:
    return Graph({(get_stack_subframe_image, stack): {}})
  else:
    return Graph({(stack_files, stack): {(build_stacking_subgraph, stack)}})
