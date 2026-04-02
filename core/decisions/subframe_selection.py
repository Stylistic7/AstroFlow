import structlog
from dataclasses import replace
from frozendict import frozendict

from stacks.composite_stack import Stack
from stacks.run import RunsDict
from metadata.classes import ImageMetadata, MetadataDict
from decisions.decisions import (
    DecisionPoint, DecisionType, DecisionOverride, DecisionKey,
    ObjectType, resolve_decision,
)

log = structlog.get_logger()

_DEFAULT_QUALITY_THRESHOLD = 0.75
_DECISION_SIGNIFICANCE_THRESHOLD = 0.25  # >25% excluded → generate decision


def get_light_frames_sorted_by_stars(
    stack: Stack,
) -> list[ImageMetadata]:
    metadata = stack.all_metadata()
    assert all(f in metadata for f in stack.all_frames())
    return sorted(
        metadata,
        key=lambda i: metadata[i].plate_solution.num_stars,
        reverse=True,
    )


def _compute_threshold_stars(metadata, all_frames, threshold: float) -> int:
    """Compute the star count cutoff from a fractional threshold."""
    star_counts = [metadata[f].plate_solution.num_stars for f in all_frames]
    if not star_counts:
        return 0
    return int(threshold * max(star_counts))


def select_subframes(
    stack: Stack,
    min_frames_in_stack: int,
    decision_overrides: frozendict[DecisionKey, DecisionOverride] = frozendict(),
) -> Stack:
    metadata = stack.all_metadata()
    all_frames = stack.all_frames()
    total_frames = len(all_frames)

    if total_frames == 0:
        return None

    highest_star_count = max(
        metadata[f].plate_solution.num_stars for f in all_frames
    )

    log.info("Subframe selection",
             stack=stack.stable_hash[:8],
             best_stars=highest_star_count,
             total_frames=total_frames)

    # Compute what the default threshold would exclude.
    default_threshold_stars = _compute_threshold_stars(
        metadata, all_frames, _DEFAULT_QUALITY_THRESHOLD
    )
    default_excluded = sum(
        1 for f in all_frames
        if metadata[f].plate_solution.num_stars < default_threshold_stars
    )

    # Only publish a decision when the exclusion is significant.
    if total_frames > 0 and default_excluded / total_frames > _DECISION_SIGNIFICANCE_THRESHOLD:
        worst_included = min(
            (metadata[f].plate_solution.num_stars for f in all_frames
             if metadata[f].plate_solution.num_stars >= default_threshold_stars),
            default=0,
        )
        worst_excluded = min(
            (metadata[f].plate_solution.num_stars for f in all_frames
             if metadata[f].plate_solution.num_stars < default_threshold_stars),
            default=0,
        )

        decision = DecisionPoint.make(
            decision_type=DecisionType.SUBFRAME_QUALITY,
            object_id=stack.stable_hash,
            subject_id=stack.stable_hash,
            title=f"{default_excluded} of {total_frames} frames excluded by quality filter",
            description=(
                f"The quality filter (≥{_DEFAULT_QUALITY_THRESHOLD:.0%} of best star count) "
                f"would exclude {default_excluded} of {total_frames} frames. "
                f"Best frame: {highest_star_count} stars, threshold: {default_threshold_stars} stars. "
                f"Override with 0.0 to keep all frames, or a different threshold (0.0–1.0)."
            ),
            metadata={
                "default_threshold": _DEFAULT_QUALITY_THRESHOLD,
                "total_frames": total_frames,
                "excluded_count": default_excluded,
                "best_star_count": highest_star_count,
                "threshold_stars": default_threshold_stars,
                "worst_included_stars": worst_included,
                "worst_excluded_stars": worst_excluded,
            },
        )

        # The resolved value is the quality threshold to use (a float).
        # Default: _DEFAULT_QUALITY_THRESHOLD. Override: user-provided float.
        quality_threshold = resolve_decision(
            decision, decision_overrides, default=_DEFAULT_QUALITY_THRESHOLD
        )
    else:
        quality_threshold = _DEFAULT_QUALITY_THRESHOLD

    # Apply the resolved threshold.
    threshold_stars = _compute_threshold_stars(metadata, all_frames, quality_threshold)
    selected_light_frames = [
        f for f in all_frames
        if metadata[f].plate_solution.num_stars >= threshold_stars
    ]
    excluded_count = total_frames - len(selected_light_frames)

    log.info("Subframe quality filter",
             threshold=quality_threshold,
             threshold_stars=threshold_stars,
             selected=len(selected_light_frames),
             excluded=excluded_count)

    if len(selected_light_frames) < min_frames_in_stack:
        return None

    new_contents = RunsDict()
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
