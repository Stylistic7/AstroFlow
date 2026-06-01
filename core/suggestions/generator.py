import os
from math import ceil
from pathlib import Path
from statistics import median

import structlog

from context import context
from decisions.subframe_selection import effective_integration
from metadata.classes import ImageMetadata
from metadata.sidecar_io import read_image_metadata
from stacks.composite_stack import Stack

from suggestions.types import Suggestion, SuggestionKind

log = structlog.get_logger()


PER_FILTER_PRESET_S: dict[str, int] = {
    "H": 300, "S": 300, "O": 300,
    "L": 120,
    "R": 60, "G": 60, "B": 60,
}
DEFAULT_PRESET_S: int = 120
FILTER_PRIORITY: tuple[str, ...] = ("H", "S", "O", "L", "R", "G", "B")
COMPLETION_THRESHOLD: float = 0.9  # emit COMPLETION when current < 90% of peer max


def _captured_exposures(stack: Stack) -> list[float]:
    """Raw per-frame EXPOSURE values across every captured frame.

    Used by `_per_frame_default` to suggest a future capture cadence —
    a question about the user's CAPTURE habits, NOT about which frames
    a stacker keeps. Deliberately unfiltered: a low-star-count frame
    still tells you what cadence the user typically shoots at.
    """
    metadata = stack.all_metadata()
    return [
        float(metadata[f].header["EXPOSURE"])
        for f in stack.all_frames()
        if "EXPOSURE" in metadata[f].header
    ]


def _per_frame_default(per_frame_list: list[float], filter: str) -> float:
    if per_frame_list:
        rounded = float(round(median(per_frame_list)))
        if rounded > 0:
            return rounded
    return float(PER_FILTER_PRESET_S.get(filter, DEFAULT_PRESET_S))


def _master_meta_by_stack(
    cart_item: tuple[str, str, str],
) -> dict[str, ImageMetadata]:
    """For a `(object_type, object_id, pipeline_name)` cart entry, return
    `{stack.stable_hash: ImageMetadata}` for every stack whose master
    exists on disk under this pipeline. One trace round-trip; pure
    in-memory mapping afterward.

    Stacks without a built master, or whose pipeline output isn't
    resolvable yet, are absent from the mapping — caller falls back
    to AstroFlow's predicted-effective for those.
    """
    from dependency_graph.trace_paths import traced_paths_for

    try:
        trace = traced_paths_for(cart_item)
    except Exception as e:
        log.warning("trace failed; using predicted-effective fallback",
                    cart_item=cart_item, error=str(e))
        return {}

    out: dict[str, ImageMetadata] = {}
    for entry in trace:
        if entry["op_name"] != "stack_files":
            continue
        stack_hash = entry["identity"].get("stack")
        if not stack_hash or stack_hash in out:
            continue
        path = entry["path"]
        if not os.path.exists(path):
            continue
        meta = read_image_metadata(Path(path))
        if meta is not None:
            out[stack_hash] = meta
    return out


def generate_suggestions(allowed_filters: frozenset[str]) -> list[Suggestion]:
    """Build suggestion list from Shopping Cart mosaics.

    Pure read — no writes to context, no decisions published.

    Per-stack effective integration prefers the stacker's own
    post-rejection number (via `effective_integration` consulting the
    master's `ImageMetadata.master_integration_s`); falls back to
    AstroFlow's quality-filter prediction when no master is available.
    Per-frame cadence is the raw EXPOSURE median across all captured
    frames — a property of capture habits, not stacker rejection.
    """
    cart = context.shopping_cart or []
    # One mosaic_id -> (cart_item) map. If the user has the same mosaic
    # in the cart under multiple pipelines (rare), the first entry wins
    # for master lookup; the trade-off is documented in the plan.
    mosaic_cart_item: dict[str, tuple[str, str, str]] = {}
    for entry in cart:
        if not (isinstance(entry, (list, tuple)) and len(entry) >= 3):
            continue
        if entry[0] != "mosaic":
            continue
        mosaic_cart_item.setdefault(entry[1], tuple(entry[:3]))

    mosaics_dict = context.mosaics or {}
    mosaics = {
        mid: ref.get()
        for mid, ref in mosaics_dict.items()
        if mid in mosaic_cart_item
    }
    targets = context.targets or {}
    stacks = context.stacks or {}

    suggestions: list[Suggestion] = []

    for mosaic_id, mosaic in mosaics.items():
        master_meta_by_stack = _master_meta_by_stack(mosaic_cart_item[mosaic_id])

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
                total_s = effective_integration(
                    stack,
                    master_meta=master_meta_by_stack.get(stack.stable_hash),
                )
                if total_s <= 0:
                    continue
                per_frame = _captured_exposures(stack)
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
