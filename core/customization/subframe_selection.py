import structlog
from dataclasses import replace
from frozendict import frozendict

from stacks.composite_stack import CompositeStack
from stacks.simple_stack import SimpleStacksDict
from metadata.classes import ImageMetadata, MetadataDict
from customization.decisions import (
    DecisionPoint, DecisionType, DecisionOverride, DecisionKey,
    BinaryChoice, ObjectType, resolve_decision,
)

log = structlog.get_logger()

_QUALITY_THRESHOLD = 0.75
_DECISION_SIGNIFICANCE_THRESHOLD = 0.25  # >25% excluded → generate decision


def get_light_frames_sorted_by_stars(
    stack: CompositeStack,
) -> list[ImageMetadata]:
    metadata = stack.all_metadata()
    assert all(f in metadata for f in stack.all_frames())
    return sorted(
        metadata,
        key=lambda i: metadata[i].plate_solution.num_stars,
        reverse=True,
    )


def select_subframes(
    stack: CompositeStack,
    min_frames_in_stack: int,
    decision_overrides: frozendict[DecisionKey, DecisionOverride] = frozendict(),
) -> CompositeStack:
    metadata = stack.all_metadata()
    sorted_light_frames = get_light_frames_sorted_by_stars(stack)
    total_frames = len(sorted_light_frames)
    highest_star_count = metadata[sorted_light_frames[0]].plate_solution.num_stars

    log.info("Subframe selection",
             stack=stack.stable_hash[:8],
             best_stars=highest_star_count,
             total_frames=total_frames)

    threshold_stars = int(_QUALITY_THRESHOLD * highest_star_count)
    selected_light_frames = [
        f for f in stack.all_frames()
        if metadata[f].plate_solution.num_stars >= threshold_stars
    ]
    excluded_count = total_frames - len(selected_light_frames)

    # Publish a decision (if significant) and resolve it.
    # resolve_decision handles publishing to context if not already there.
    if total_frames > 0 and excluded_count / total_frames > _DECISION_SIGNIFICANCE_THRESHOLD:
        worst_included = min(
            metadata[f].plate_solution.num_stars for f in selected_light_frames
        ) if selected_light_frames else 0
        worst_excluded = min(
            metadata[f].plate_solution.num_stars
            for f in stack.all_frames()
            if f not in selected_light_frames
        ) if excluded_count > 0 else 0

        decision = DecisionPoint(
            key=(DecisionType.SUBFRAME_QUALITY, ObjectType.STACK,
                 stack.stable_hash, stack.stable_hash),
            decision_type=DecisionType.SUBFRAME_QUALITY,
            default_choice=BinaryChoice.INCLUDE,
            title=f"{excluded_count} of {total_frames} frames excluded by quality filter",
            description=(
                f"The quality filter (≥{_QUALITY_THRESHOLD:.0%} of best star count) "
                f"would exclude {excluded_count} of {total_frames} frames. "
                f"Best frame: {highest_star_count} stars, threshold: {threshold_stars} stars. "
                f"Override to EXCLUDE to skip the filter and keep all frames."
            ),
            metadata=frozendict({
                "threshold": _QUALITY_THRESHOLD,
                "total_frames": total_frames,
                "excluded_count": excluded_count,
                "best_star_count": highest_star_count,
                "threshold_stars": threshold_stars,
                "worst_included_stars": worst_included,
                "worst_excluded_stars": worst_excluded,
            }),
            object_type=ObjectType.STACK,
            object_id=stack.stable_hash,
            subject_id=stack.stable_hash,
        )

        choice = resolve_decision(decision, decision_overrides, default=BinaryChoice.INCLUDE)
        if choice == BinaryChoice.EXCLUDE:
            log.info("Quality filter skipped by decision override")
            return stack

    log.info("Subframe quality filter",
             selected=len(selected_light_frames),
             excluded=excluded_count,
             threshold_stars=threshold_stars)

    if len(selected_light_frames) < min_frames_in_stack:
        return None

    new_contents = SimpleStacksDict()
    for content_hash, substack in stack.contents.items():
        if new_frames := tuple(
            f for f in substack.frames if f in selected_light_frames
        ):
            new_metadata = MetadataDict({
                h: m for h, m in metadata.items() if h in new_frames
            })
            new_substack = replace(
                substack, frames=new_frames, metadata=new_metadata
            )
            new_contents[content_hash] = new_substack

    return replace(stack, contents=new_contents)
