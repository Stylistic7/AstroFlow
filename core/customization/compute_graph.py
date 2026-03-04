import ray
from dataclasses import replace

from context import context
from stacks.types import StackHash
from stacks.composite_stack import CompositeStack, CompositeStacksDict
from stacks.simple_stack import SimpleStacksDict
from calibration import associate_with_stack
from decorators import cached
from customization.subframe_selection import select_subframes
from customization.settings import Settings, StackSettings
from customization.types import CustomizedStack, CustomizedStacksDict
from quarantining.quarantining import apply_quarantine
from mosaics.types import MosaicsDict
from targets.types import TargetsDict


def apply_settings(stack: CompositeStack, settings: Settings) -> CompositeStack:
    new_contents = {}
    for content_hash, substack in stack.contents.items():
        substack_settings = settings.stack_settings.get(
            substack.content_hash, StackSettings()
        )
        new_dependencies = (
            substack.dependencies | substack_settings.dependencies
        )
        new_substack = replace(substack, dependencies=new_dependencies)
        new_contents[content_hash] = new_substack

    return replace(stack, contents=new_contents)


@ray.remote
def store_customized_stack(stack) -> None:
    context.or_equals("stacks", {stack.stable_hash: stack})


# TODO: Add tests in test_customization.py
def customize_stack(
    stack: CompositeStack,
    cal_stacks: SimpleStacksDict,
    settings: Settings,
    quarantined_stacks: tuple[StackHash],
) -> CustomizedStack:
    def chain(x, *funcs):
        for f in funcs:
            if x is None:
                return None
            x = f(x)
        return x

    stack = chain(
        stack,
        lambda s: associate_with_stack(s, cal_stacks),
        select_subframes,
        lambda s: apply_quarantine(s, quarantined_stacks),
        lambda s: apply_settings(s, settings),
    )

    if stack is None:
        return None

    return CustomizedStack(**stack.__dict__)


@cached.remote
def customize_stacks(
    settings: Settings,
    quarantined_stacks: tuple[StackHash],
    stacks: CompositeStacksDict,
    cal_stacks: SimpleStacksDict,
) -> CustomizedStacksDict:
    result = {
        k: customize_stack(v, cal_stacks, settings, quarantined_stacks)
        for k, v in stacks.items()
    }
    result = {k: v for k, v in result.items() if v is not None}
    return result


# TODO: Implement real customization features.
@cached.remote
def scatter_customize_targets(
    settings: Settings, targets: TargetsDict
) -> TargetsDict:
    return targets


# TODO: Implement real customization features.
@cached.remote
def scatter_customize_mosaics(
    settings: Settings, mosaics: MosaicsDict
) -> MosaicsDict:
    return mosaics


@cached.remote
def customization_done() -> None:
    pass
