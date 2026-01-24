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
from customization.types import CustomizedStack, CustomizedStacksDict
from quarantining.quarantining import apply_quarantine
from mosaics.types import MosaicsDict, MosaicHash
from targets.types import TargetsDict


def apply_settings(stack: CompositeStack, settings: Settings) -> CompositeStack:
    new_contents = {}
    for content_hash, substack in stack.contents.items():
        substack_settings = settings.stack_settings.get(
            substack.content_hash, StackSettings()
        )
        new_dependencies = substack.dependencies | substack_settings.dependencies
        new_substack = replace(substack, dependencies=new_dependencies)
        new_contents[content_hash] = new_substack

    return replace(stack, contents=new_contents)


@cached.remote
def gather_customize_stacks(*stacks) -> CustomizedStacksDict:
    print(f"### gather_customize_stacks type(stacks) {type(stacks)} len {len(stacks)}")
    return {s.stable_hash: s for s in stacks if s is not None}


@ray.remote
def store_customized_stack(stack) -> None:
    context.or_equals("stacks", {stack.stable_hash: stack})


@cached.remote
def scatter_customize_stacks(
    settings: Settings, stacks: CompositeStacksDict, cal_stacks: SimpleStacksDict
) -> Graph:
    print(f"### scatter_customize_stacks type(stacks) {type(stacks)} len {len(stacks)}")
    print(
        f"### scatter_customize_stacks type(cal_stacks) {type(cal_stacks)} len {len(cal_stacks)}"
    )

    tasks = []
    for stack_id, stack in stacks.items():
        tasks.append((customize_stack, stack, cal_stacks, settings))

    return Graph({gather_customize_stacks: tasks})


# TODO: Add tests in test_customization.py
# TODO: Pass in the quarantined_stacks and cache this.
@cached.remote
def customize_stack(
    stack: CompositeStack, cal_stacks: SimpleStacksDict, settings: Settings
) -> CustomizedStack:
    stack = associate_with_stack(stack, cal_stacks)
    stack = select_subframes(stack)
    if stack is None:
        print(f"### Stack eliminated by subframe selection")
        return None

    stack = apply_quarantine(stack)
    stack = apply_settings(stack, settings)

    return CustomizedStack(**stack.__dict__)


# TODO: Implement real customization features.
@cached.remote
def scatter_customize_targets(settings: Settings, targets: TargetsDict) -> TargetsDict:
    return targets


# TODO: Implement real customization features.
@cached.remote
def scatter_customize_mosaics(settings: Settings, mosaics: MosaicsDict) -> MosaicsDict:
    return mosaics


@cached.remote
def customization_done() -> None:
    pass
