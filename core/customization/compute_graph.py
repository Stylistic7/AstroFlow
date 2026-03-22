import ray
from dataclasses import replace
from beartype import beartype
from frozendict import frozendict

from context import context
from cache.context_ref import ContextRef
from stacks.composite_stack import CompositeStack, CompositeStacksDict
from stacks.simple_stack import SimpleStack, SimpleStacksDict
from calibration import associate_with_stack
from decorators import cached
from dependency_graph import Graph
from customization.subframe_selection import select_subframes
from customization.settings import Settings, StackSettings, TargetSettings
from customization.types import CustomizedStack, CustomizedStacksDict
from customization.decisions import BinaryChoice, DecisionKey, DecisionOverride, ObjectType
from quarantining.quarantining import apply_quarantine
from mosaics.types import MosaicsDict
from targets.types import TargetsDict


def apply_decision_exclusions(
    stack: CompositeStack,
    decision_overrides: frozendict[DecisionKey, DecisionOverride],
) -> CompositeStack | None:
    """Remove substacks that have been explicitly excluded via decision overrides."""
    excluded_substacks = {
        key[3]  # subject_id (the substack hash)
        for key, override in decision_overrides.items()
        if (
            key[1] == ObjectType.STACK
            and override.choice == BinaryChoice.EXCLUDE
        )
    }
    if not excluded_substacks:
        return stack
    return apply_quarantine(stack, excluded_substacks)


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
@beartype
def store_customized_stack(stack: CustomizedStack) -> None:
    new_ref = ContextRef.store(("stacks", stack.stable_hash), stack)
    existing = context.stacks or CompositeStacksDict()
    merged = CompositeStacksDict({**existing, stack.stable_hash: new_ref})
    context.stacks = merged


# TODO: Add tests in test_customization.py
@cached.remote
def customize_stack(
    stack: CompositeStack,
    cal_stacks: SimpleStacksDict,
    stack_settings: StackSettings,
    decision_overrides: frozendict,
    min_frames_in_stack: int,
) -> CustomizedStack | None:
    def chain(x, *funcs):
        for f in funcs:
            if x is None:
                return None
            x = f(x)
        return x

    stack = chain(
        stack,
        lambda s: associate_with_stack(s, cal_stacks),
        lambda s: select_subframes(s, min_frames_in_stack),
        lambda s: apply_quarantine(s, stack_settings.quarantined_substacks),
        lambda s: apply_decision_exclusions(s, decision_overrides),
        lambda s: apply_settings(s, stack_settings),
    )

    if stack is None:
        return None

    return CustomizedStack(**stack.__dict__)


@ray.remote
@beartype
def gather_customize_stacks(*ancestor_results: CustomizedStack | None) -> CustomizedStacksDict:
    return CustomizedStacksDict({
        s.stable_hash: ContextRef.store(("stacks", s.stable_hash), s)
        for s in ancestor_results if s is not None
    })


@cached.remote
def scatter_customize_stacks(
    stack_settings: frozendict,
    decision_overrides: frozendict,
    min_frames_in_stack: int,
    stacks: CompositeStacksDict,
    cal_stacks: SimpleStacksDict,
) -> Graph:
    print(f"### scatter_customize_stacks type(cal_stacks) {type(cal_stacks)}")

    def _overrides_for_stack(stack_hash):
        """Extract only the decision overrides relevant to this stack."""
        return frozendict({
            k: v for k, v in decision_overrides.items()
            if len(k) >= 3 and k[2] == stack_hash
        })

    return Graph(
        {
            gather_customize_stacks: [
                (
                    customize_stack,
                    ref.get(),
                    cal_stacks,
                    stack_settings.get(k, StackSettings()),
                    _overrides_for_stack(k),
                    min_frames_in_stack,
                )
                for k, ref in stacks.items()
            ]
        }
    )


# TODO: Implement real customization features.
@cached.remote
def scatter_customize_cal_stacks(cal_stacks: SimpleStacksDict) -> SimpleStacksDict:
    print(
        f"### scatter_customize_cal_stacks type(cal_stacks) {type(cal_stacks)}"
    )
    # cal_stacks is a SimpleStacksDict of real SimpleStack objects — pass through as-is
    return cal_stacks


@cached.remote
def scatter_customize_targets(
    target_settings: frozendict, targets: TargetsDict
) -> TargetsDict:
    return TargetsDict(
        {
            h: ref
            for h, ref in targets.items()
            if not target_settings.get(h, TargetSettings()).is_quarantined
        }
    )


# TODO: Implement real customization features.
@cached.remote
def scatter_customize_mosaics(
    mosaic_settings: frozendict, mosaics: MosaicsDict
) -> MosaicsDict:
    return mosaics
