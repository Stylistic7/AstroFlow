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

# Keep ALL subframes by default. A user dials this down per-Stack via a
# subframe_quality DecisionOverride (0.0–1.0 fraction of the best
# frame's star count); only then are frames dropped.
_DEFAULT_QUALITY_THRESHOLD = 0.0
_DECISION_SIGNIFICANCE_THRESHOLD = 0.25  # >25% excluded → generate decision


def resolved_quality_threshold(
    stack: Stack,
    decision_overrides: "frozendict[DecisionKey, DecisionOverride]" = frozendict(),
) -> float:
    """The subframe-quality threshold actually in effect for this stack:
    a per-Stack user override if present, else the default (0.0 = keep
    all). Single source of truth, side-effect-free — `select_subframes`
    and the UI's per-run counts must agree, so both go through here.
    """
    override = decision_overrides.get(
        DecisionKey.subframe_quality(stack.stable_hash)
    )
    return (
        float(override.choice) if override is not None
        else _DEFAULT_QUALITY_THRESHOLD
    )


def selected_counts_by_run(
    stack: Stack,
    decision_overrides: "frozendict[DecisionKey, DecisionOverride]" = frozendict(),
) -> dict[str, tuple[int, int]]:
    """Per run content_hash → (selected, total) frame counts under the
    resolved subframe-quality threshold. Side-effect-free; mirrors
    `select_subframes`' inclusion math (a stack-global star cutoff,
    applied per run) so the RunsTab shows exactly what stacking uses.
    """
    metadata = stack.all_metadata()
    all_frames = stack.all_frames()
    threshold_stars = _compute_threshold_stars(
        metadata, all_frames,
        resolved_quality_threshold(stack, decision_overrides),
    )
    counts: dict[str, tuple[int, int]] = {}
    for content_hash, run in stack.contents.items():
        selected = sum(
            1 for f in run.frames
            if metadata[f].plate_solution.num_stars >= threshold_stars
        )
        counts[content_hash] = (selected, len(run.frames))
    return counts


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


def predicted_effective_integration(stack: Stack) -> float:
    """AstroFlow's PRE-STACKING prediction of usable integration: the
    sum of EXPOSURE across frames that pass the default subframe-quality
    filter. Side-effect-free; mirrors `select_subframes`' inclusion math
    without publishing decisions or applying overrides.

    Distinct from `effective_integration`, which prefers the stacker's
    own post-rejection number when a master is available — this is the
    fallback used when no master exists yet (or the stacker doesn't
    expose a survivor total).
    """
    metadata = stack.all_metadata()
    all_frames = stack.all_frames()
    if not all_frames:
        return 0.0
    threshold_stars = _compute_threshold_stars(
        metadata, all_frames, _DEFAULT_QUALITY_THRESHOLD
    )
    return float(sum(
        metadata[f].header.get("EXPOSURE", 0)
        for f in all_frames
        if metadata[f].plate_solution.num_stars >= threshold_stars
    ))


def effective_integration(
    stack: Stack, master_meta: ImageMetadata | None = None,
) -> float:
    """How much usable signal does this stack have?

    Prefers the stacker's own post-rejection integration time when a
    master ImageMetadata is available with `master_integration_s`
    populated (e.g., PixInsight's `PCL:TotalExposureTime`). Otherwise
    falls back to AstroFlow's pre-stacking prediction. One canonical
    accessor so every consumer — API `integration` field, suggestion
    completion bar, dual-exposure widget — agrees on a single number.
    """
    if master_meta is not None and master_meta.master_integration_s is not None:
        return float(master_meta.master_integration_s)
    return predicted_effective_integration(stack)


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
            key=DecisionKey.subframe_quality(stack.stable_hash),
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
        # Honor a per-Stack override even when the default (keep-all)
        # excludes nothing — without this, dialing down a specific
        # Stack would be silently ignored, since the decision is only
        # published/resolved above when the DEFAULT is significant.
        quality_threshold = resolved_quality_threshold(
            stack, decision_overrides
        )

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
