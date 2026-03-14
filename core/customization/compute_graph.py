import ray
from dataclasses import replace
from frozendict import frozendict

from context import context
from stacks.composite_stack import CompositeStack
from stacks.simple_stack import SimpleStack, SimpleStacksDict
from calibration import associate_with_stack
from decorators import cached
from dependency_graph import Graph
from customization.subframe_selection import select_subframes
from customization.settings import Settings, StackSettings, TargetSettings
from customization.types import CustomizedStack, CustomizedStacksDict
from quarantining.quarantining import apply_quarantine
from mosaics.types import MosaicsDict
from targets.types import TargetsDict


def apply_settings(
    stack: CompositeStack, stack_settings: StackSettings
) -> CompositeStack:
    new_contents = SimpleStacksDict()
    for content_hash, substack in stack.contents.items():
        new_dependencies = substack.dependencies | stack_settings.dependencies
        new_substack = replace(substack, dependencies=new_dependencies)
        new_contents[content_hash] = new_substack

    return replace(stack, contents=new_contents)


@ray.remote
def store_customized_stack(stack) -> None:
    context.or_equals("stacks", {stack.stable_hash: stack})


# TODO: Add tests in test_customization.py
@cached.remote
def customize_stack(
    stack: CompositeStack,
    cal_stacks: SimpleStacksDict,
    settings: StackSettings,
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
        lambda s: apply_quarantine(s, settings.quarantined_substacks),
        lambda s: apply_settings(s, settings),
    )

    if stack is None:
        return None

    return CustomizedStack(**stack.__dict__)


@ray.remote
def gather_customize_stacks(*ancestor_results) -> None:
    return {s.stable_hash: s for s in ancestor_results if s is not None}


@cached.remote
def scatter_customize_stacks(
    settings: Settings,
    stacks: CustomizedStacksDict,
    cal_stacks: SimpleStacksDict,
) -> CustomizedStacksDict:
    print(f"### scatter_customize_stacks type(cal_stacks) {type(cal_stacks)}")
    return Graph(
        {
            gather_customize_stacks: [
                (
                    customize_stack,
                    v,
                    frozendict(cal_stacks),
                    settings.stack_settings.get(k, StackSettings()),
                )
                for k, v in stacks.items()
            ]
        }
    )


# TODO: Implement real customization features.
@cached.remote
def scatter_customize_cal_stacks(
    settings: Settings, cal_stacks: SimpleStacksDict
) -> SimpleStacksDict:
    print(
        f"### scatter_customize_cal_stacks type(cal_stacks) {type(cal_stacks)}"
    )
    return cal_stacks


@cached.remote
def scatter_customize_targets(
    settings: Settings, targets: TargetsDict
) -> TargetsDict:
    return TargetsDict(
        {
            h: t
            for h, t in targets.items()
            if not settings.target_settings.get(h, TargetSettings()).is_quarantined
        }
    )


# TODO: Implement real customization features.
@cached.remote
def scatter_customize_mosaics(
    settings: Settings, mosaics: MosaicsDict
) -> MosaicsDict:
    return mosaics
