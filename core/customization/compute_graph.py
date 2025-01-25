from dataclasses import replace

from context import context
from dependency_graph import Graph
from stacks.composite_stack import CompositeStack
from calibration import associate_with_stack
from dependency_graph.astroflow_node import AstroFlowNode
from customization.subframe_selection import select_subframes
from customization.settings import Settings, StackSettings
from customization.types import CustomizedStack


def apply_settings(stack: CompositeStack) -> CompositeStack:
  settings = context.settings or Settings()
  new_contents = {}
  for content_hash, substack in stack.contents.items():
    substack_settings = settings.stack_settings.get(substack.content_hash,
                                                    StackSettings())
    new_dependencies = substack.dependencies | substack_settings.dependencies
    new_substack = replace(substack, dependencies=new_dependencies)
    new_contents[content_hash] = new_substack

  return replace(stack, contents=new_contents)


@AstroFlowNode()
def gather_customize_stacks(*stacks) -> None:
  context.stacks = {s.stable_hash: s for s in stacks}


@AstroFlowNode()
def scatter_customize_stacks() -> Graph:
  tasks = []
  for stack in context.grouped_stacks.values():
    tasks.append((customize_stack, stack))

  return Graph({gather_customize_stacks: tasks})


# TODO: Add tests in test_customization.py
@AstroFlowNode()
def customize_stack(stack: CompositeStack) -> CustomizedStack:
  stack = associate_with_stack(stack)
  stack = apply_settings(stack)

  # TODO: Bug fix: subframe selection blows away associations...
  # stack = select_subframes(stack)

  return CustomizedStack(**stack.__dict__)


@AstroFlowNode()
def scatter_customize_targets() -> None:
  # TODO: Implement real customization features.
  context.targets = context.grouped_targets


@AstroFlowNode()
def scatter_customize_mosaics(*mosaics) -> None:
  # TODO: Implement real customization features.
  context.mosaics = context.grouped_mosaics


@AstroFlowNode()
def customization_done() -> None:
  pass


@AstroFlowNode()
def build_customize_graph() -> Graph:
  return Graph({
      customization_done: [
          scatter_customize_stacks, scatter_customize_targets,
          scatter_customize_mosaics
      ]
  })
