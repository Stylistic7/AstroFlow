from dataclasses import replace

from context import context
from dependency_graph import Graph
from stacks.composite_stack import CompositeStack
from calibration import associate_with_stack
from task.astroflow_node import task
from customization.subframe_selection import select_subframes
from customization.settings import Settings, StackSettings
from customization.types import CustomizedStack
from quarantining.quarantining import apply_quarantine


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


@task.remote
def gather_customize_stacks(*stacks) -> None:
  context.stacks = {s.stable_hash: s for s in stacks}


@task.remote
def scatter_customize_stacks() -> Graph:
  quarantined_stacks = context.quarantined_stacks or []

  tasks = []
  for stack_id, stack in context.grouped_stacks.items():
    if stack_id not in quarantined_stacks:
      tasks.append((customize_stack, stack))

  return Graph({gather_customize_stacks: tasks})


# TODO: Add tests in test_customization.py
# TODO: Pass in the StackSettings and cache this.
# TODO: Pass in the quarantined_stacks and cache this.
@task.remote
def customize_stack(stack: CompositeStack) -> CustomizedStack:
  stack = associate_with_stack(stack)
  stack = select_subframes(stack)
  stack = apply_quarantine(stack)
  stack = apply_settings(stack)

  return CustomizedStack(**stack.__dict__)


@task.remote
def scatter_customize_targets() -> None:
  # TODO: Implement real customization features.
  quarantined_targets = context.quarantined_targets or []
  context.targets = {
      id: t
      for id, t in context.grouped_targets.items()
      if id not in quarantined_targets
  }


@task.remote
def scatter_customize_mosaics(*mosaics) -> None:
  # TODO: Implement real customization features.
  quarantined_mosaics = context.quarantined_mosaics or []
  context.mosaics = {
      id: t
      for id, t in context.grouped_mosaics.items()
      if id not in quarantined_mosaics
  }


@task.remote
def customization_done() -> None:
  pass


@task.remote
def build_customize_graph() -> Graph:
  return Graph({
      customization_done: [
          scatter_customize_stacks, scatter_customize_targets,
          scatter_customize_mosaics
      ]
  })
