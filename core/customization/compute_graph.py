import ray
import structlog
from dataclasses import replace
from beartype import beartype
from frozendict import frozendict

from context import context
from cache.context_ref import ContextRef
from stacks.composite_stack import Stack, StacksDict, CalibrationStacksDict
from stacks.run import Run, RunsDict
from calibration import associate_with_stack
from decorators import node
from dependency_graph import Graph
from customization.settings import Settings, StackSettings, TargetSettings
from customization.decisions import DecisionType
from customization.resolve import resolve_decisions as _resolve_decisions
from mosaics.types import MosaicsDict
from targets.types import TargetsDict

log = structlog.get_logger()

# All exclusion decision types, applied in precedence order by resolve_decisions.
ALL_EXCLUSION_TYPES = (
    DecisionType.QUARANTINE,
    DecisionType.ROTATION_OUTLIER,
    DecisionType.SUBFRAME_QUALITY,
)


def apply_settings(
    stack: Stack, stack_settings: StackSettings
) -> Stack:
    new_contents = RunsDict()
    for content_hash, substack in stack.contents.items():
        new_dependencies = substack.dependencies | stack_settings.dependencies
        new_substack = replace(substack, dependencies=new_dependencies)
        new_contents[content_hash] = new_substack

    return replace(stack, contents=new_contents)


@node
def enrich_with_calibration(
    stack_ref: ContextRef[Stack],
    cal_stacks: CalibrationStacksDict,
    stack_settings: StackSettings,
    stack_overrides: frozendict,
    min_frames_in_stack: int,
) -> Stack:
    """Enrich stack with calibration deps and user settings. All frames preserved.

    Also detects and publishes decisions (quality, rotation) as side effects
    so they're visible in the UI, but does NOT filter frames.
    """
    import time
    t0 = time.time()
    stack = stack_ref.get()
    print(f"### enrich_with_calibration: stack={stack.stable_hash[:8]} frames={len(stack.all_frames())} cal_stacks={len(cal_stacks)}")
    t1 = time.time()
    stack = associate_with_stack(stack, cal_stacks)
    print(f"### enrich_with_calibration: associate_with_stack took {time.time()-t1:.3f}s")
    t1 = time.time()
    stack = apply_settings(stack, stack_settings)
    print(f"### enrich_with_calibration: apply_settings took {time.time()-t1:.3f}s")

    # Detect and publish decisions without filtering.
    # resolve_decisions publishes DecisionPoints to context as a side effect.
    t1 = time.time()
    _resolve_decisions(stack, ALL_EXCLUSION_TYPES, stack_overrides, min_frames_in_stack)
    print(f"### enrich_with_calibration: resolve_decisions took {time.time()-t1:.3f}s")
    print(f"### enrich_with_calibration: total {time.time()-t0:.3f}s")

    return stack




@ray.remote
@beartype
def gather_enriched_stacks(*ancestor_results: Stack) -> StacksDict:
    return StacksDict({
        s.stable_hash: ContextRef.store(("stacks", s.stable_hash), s)
        for s in ancestor_results if s is not None
    })



@node
def scatter_customize_stacks(
    stack_settings: frozendict,
    decision_overrides: frozendict,
    min_frames_in_stack: int,
    stacks: StacksDict,
    cal_stacks: CalibrationStacksDict,
) -> Graph:
    log.debug("Scattering stack customization", cal_stacks_type=type(cal_stacks).__name__)

    def _overrides_for_stack(stack_hash):
        """Extract only the decision overrides relevant to this stack."""
        return frozendict({
            k: v for k, v in decision_overrides.items()
            if len(k) >= 3 and k[2] == stack_hash
        })

    enrich_tasks = [
        (
            enrich_with_calibration,
            ref,
            cal_stacks,
            stack_settings.get(k, StackSettings()),
            _overrides_for_stack(k),
            min_frames_in_stack,
        )
        for k, ref in stacks.items()
    ]

    return Graph({gather_enriched_stacks: enrich_tasks})


# TODO: Implement real customization features.
@node
def scatter_customize_cal_stacks(cal_stacks: CalibrationStacksDict) -> CalibrationStacksDict:
    log.debug("Scattering calibration stack customization", cal_stacks_type=type(cal_stacks).__name__)
    return cal_stacks


@node
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
@node
def scatter_customize_mosaics(
    mosaic_settings: frozendict, mosaics: MosaicsDict
) -> MosaicsDict:
    return mosaics
