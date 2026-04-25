from math import ceil
from statistics import median

from context import context
from stacks.composite_stack import Stack
from decisions.subframe_selection import _DEFAULT_QUALITY_THRESHOLD

from suggestions.types import Suggestion, SuggestionKind


PER_FILTER_PRESET_S: dict[str, int] = {
    "H": 300, "S": 300, "O": 300,
    "L": 120,
    "R": 60, "G": 60, "B": 60,
}
DEFAULT_PRESET_S: int = 120
FILTER_PRIORITY: tuple[str, ...] = ("H", "S", "O", "L", "R", "G", "B")
COMPLETION_THRESHOLD: float = 0.9  # emit COMPLETION when current < 90% of peer max


def _surviving_exposure(stack: Stack) -> tuple[float, list[float]]:
    """Exposure of frames that survive the default subframe quality threshold.

    Side-effect-free: duplicates the filter logic from select_subframes without
    publishing decision points.

    Returns (total_seconds, [per_frame_exposure_seconds]).
    """
    metadata = stack.all_metadata()
    all_frames = stack.all_frames()
    if not all_frames:
        return 0.0, []
    star_counts = [metadata[f].plate_solution.num_stars for f in all_frames]
    if not star_counts:
        return 0.0, []
    threshold_stars = int(_DEFAULT_QUALITY_THRESHOLD * max(star_counts))
    surviving = [
        float(metadata[f].header.get("EXPOSURE", 0))
        for f in all_frames
        if metadata[f].plate_solution.num_stars >= threshold_stars
        and "EXPOSURE" in metadata[f].header
    ]
    return float(sum(surviving)), surviving


def _per_frame_default(per_frame_list: list[float], filter: str) -> float:
    if per_frame_list:
        rounded = float(round(median(per_frame_list)))
        if rounded > 0:
            return rounded
    return float(PER_FILTER_PRESET_S.get(filter, DEFAULT_PRESET_S))


def generate_suggestions(allowed_filters: frozenset[str]) -> list[Suggestion]:
    """Build suggestion list from Shopping Cart mosaics.

    Pure read — no writes to context, no decisions published.
    """
    cart = context.shopping_cart or []
    mosaic_ids_in_cart = {entry[1] for entry in cart if entry[0] == "mosaic"}
    mosaics_dict = context.mosaics or {}
    mosaics = {
        mid: ref.get()
        for mid, ref in mosaics_dict.items()
        if mid in mosaic_ids_in_cart
    }
    targets = context.targets or {}
    stacks = context.stacks or {}

    suggestions: list[Suggestion] = []

    for mosaic_id, mosaic in mosaics.items():
        # per_panel[target_id][filter] = (total_exposure_s, [per_frame_s])
        per_panel: dict[str, dict[str, tuple[float, list[float]]]] = {}
        panel_targets = {}
        for target_id in mosaic.target_ids:
            if target_id not in targets:
                continue
            target = targets[target_id].get()
            panel_targets[target_id] = target
            per_panel[target_id] = {}
            for stack_id in target.stack_ids:
                if stack_id not in stacks:
                    continue
                stack = stacks[stack_id].get()
                total_s, per_frame = _surviving_exposure(stack)
                if total_s <= 0:
                    continue
                f = stack.filter
                existing = per_panel[target_id].get(f)
                if existing is None:
                    per_panel[target_id][f] = (total_s, list(per_frame))
                else:
                    per_panel[target_id][f] = (
                        existing[0] + total_s,
                        existing[1] + per_frame,
                    )

        established_filters = {f for panel in per_panel.values() for f in panel}
        max_by_filter = {
            f: max(p[f][0] for p in per_panel.values() if f in p)
            for f in established_filters
        }

        for target_id, target in panel_targets.items():
            ra_deg, dec_deg = target.center
            panel_data = per_panel.get(target_id, {})

            for f in established_filters:
                if f not in allowed_filters:
                    continue
                target_s = max_by_filter[f]
                if f not in panel_data:
                    per_frame_s = _per_frame_default([], f)
                    count = max(1, ceil(target_s / per_frame_s)) if target_s > 0 else 1
                    suggestions.append(Suggestion(
                        kind=SuggestionKind.MISSING_CHANNEL,
                        target_id=target_id,
                        mosaic_id=mosaic_id,
                        ra_deg=float(ra_deg),
                        dec_deg=float(dec_deg),
                        filter=f,
                        current_exposure_s=0.0,
                        target_exposure_s=target_s,
                        default_exposure_per_frame_s=per_frame_s,
                        default_frame_count=count,
                        panel_label=target.name,
                        reason=(
                            f"no {f} data on {target.name} — peer panels have up to "
                            f"{target_s / 3600:.1f} h"
                        ),
                    ))
                    continue

                current_s, per_frame_list = panel_data[f]
                if target_s == 0 or current_s >= COMPLETION_THRESHOLD * target_s:
                    continue

                per_frame_s = _per_frame_default(per_frame_list, f)
                needed = target_s - current_s
                count = max(1, ceil(needed / per_frame_s))
                suggestions.append(Suggestion(
                    kind=SuggestionKind.COMPLETION,
                    target_id=target_id,
                    mosaic_id=mosaic_id,
                    ra_deg=float(ra_deg),
                    dec_deg=float(dec_deg),
                    filter=f,
                    current_exposure_s=current_s,
                    target_exposure_s=target_s,
                    default_exposure_per_frame_s=per_frame_s,
                    default_frame_count=count,
                    panel_label=target.name,
                    reason=(
                        f"{needed / 3600:.1f} h behind the best panel's {f} "
                        f"exposure ({target_s / 3600:.1f} h)"
                    ),
                ))

    filter_rank = {f: i for i, f in enumerate(FILTER_PRIORITY)}

    def sort_key(s: Suggestion):
        kind_rank = 0 if s.kind == SuggestionKind.MISSING_CHANNEL else 1
        shortfall_rank = -(s.target_exposure_s - s.current_exposure_s)
        return (kind_rank, shortfall_rank, filter_rank.get(s.filter, 99))

    suggestions.sort(key=sort_key)
    return suggestions
