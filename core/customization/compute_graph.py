import ray
from dataclasses import replace

from context import context
from dependency_graph import Graph
from stacks.composite_stack import CompositeStack, CompositeStacksDict
from stacks.simple_stack import SimpleStacksDict
from calibration import associate_with_stack
from decorators import cached
from customization.subframe_selection import select_subframes
from customization.settings import Settings, StackSettings
from customization.types import CustomizedStack
from quarantining.quarantining import apply_quarantine
from mosaics.types import MosaicsDict
from targets.types import TargetsDict

def apply_settings(stack: CompositeStack, settings: Settings) -> CompositeStack:
  new_contents = {}
  for content_hash, substack in stack.contents.items():
    substack_settings = settings.stack_settings.get(substack.content_hash,
                                                    StackSettings())
    new_dependencies = substack.dependencies | substack_settings.dependencies
    new_substack = replace(substack, dependencies=new_dependencies)
    new_contents[content_hash] = new_substack

  return replace(stack, contents=new_contents)


@ray.remote
def gather_customize_stacks(*stacks) -> None:
  print(f'### gather_customize_stacks type(stacks) {type(stacks)} len {len(stacks)}')
  stacks_dict = {s.stable_hash: s for s in stacks if s is not None}
  context.stacks = stacks_dict
  return stacks_dict


@ray.remote
def store_customized_stack(stack) -> None:
  context.or_equals('stacks', {stack.stable_hash: stack})


@ray.remote
def scatter_customize_stacks(stacks: CompositeStacksDict, cal_stacks: SimpleStacksDict) -> Graph:
  print(f'### scatter_customize_stacks type(stacks) {type(stacks)} len {len(stacks)}')
  print(f'### scatter_customize_stacks type(cal_stacks) {type(cal_stacks)} len {len(cal_stacks)}')
  
  quarantined_stacks = context.quarantined_stacks or []
  settings = context.settings or Settings()

  tasks = []
  for stack_id, stack in stacks.items():
    if stack_id not in quarantined_stacks:
      tasks.append((customize_stack, stack, cal_stacks, settings))

  return Graph({gather_customize_stacks: tasks})


# TODO: Add tests in test_customization.py
# TODO: Pass in the quarantined_stacks and cache this.
@cached.remote
def customize_stack(stack: CompositeStack,
                    cal_stacks: SimpleStacksDict,
                    settings: Settings) -> CustomizedStack:
  stack = associate_with_stack(stack, cal_stacks)
  stack = select_subframes(stack)
  if stack is None:
    print(f'### Stack eliminated by subframe selection')
    return None

  stack = apply_quarantine(stack)
  stack = apply_settings(stack, settings)

  return CustomizedStack(**stack.__dict__)


# TODO: Implement real customization features.
@ray.remote
def scatter_customize_targets(targets: TargetsDict) -> None:
  quarantined_targets = context.quarantined_targets or []
  targets_dict = {
      id: t
      for id, t in targets.items()
      if id not in quarantined_targets
  }
  context.targets = targets_dict
  print(f'### scatter_customize_targets returning {len(targets_dict)} targets')
  return targets_dict


# TODO: Implement real customization features.
@ray.remote
def scatter_customize_mosaics(mosaics: MosaicsDict) -> None:
  print(f'### scatter_customize_mosaics')
  quarantined_mosaics = context.quarantined_mosaics or []
  mosaics_dict = {
      id: t
      for id, t in mosaics.items()
      if id not in quarantined_mosaics
  }
  context.mosaics = mosaics_dict
  return mosaics_dict


@cached.remote
def customization_done() -> None:
  pass


# @ray.remote
# def build_customize_graph() -> Graph:
#   return Graph({
#       customization_done: {
#           scatter_customize_stacks, scatter_customize_targets,
#           scatter_customize_mosaics
#       }
#   })
