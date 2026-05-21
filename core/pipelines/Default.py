"""Reference pipeline: PixInsight defaults, end-to-end.

Demonstrates the whole-pipeline-as-plain-Python pattern (see the plan
at `~/.claude/plans/splendid-munching-rainbow.md`). Opt in by passing
the display name as the 4th element of a `"mosaic"` shopping-cart
entry: `("mosaic", mosaic_id, "Master", "Default")`. Bundled pipeline
names (`"PixInsight"`, `"AstroFlow"`, `"OSS"`) and registered names
(`"Default"` and any others in `~/AstroFlow/pipelines/`) live in the
same dropdown — the dispatcher routes each to the right code path.
Pipeline selection is per-cart-entry, so the same mosaic can be
dropped in twice with different pipelines for side-by-side comparison;
the choice never reaches any cached @node op's arg hash.

This reference reuses the existing PixInsight @node ops verbatim — same
function objects, same cache keys. To create a variant that swaps a
step (e.g. MSGC → GraXpert), copy this file to
`~/AstroFlow/pipelines/<YourName>.py`, change the `@pipeline("…")`
display name, and edit the call you want to replace. A future
iteration of this layer will add the inline-`@stage` recipe pattern
that lets a single file fully express its own post-processing recipe.
"""
from astroflow.ops import (
    merge_images,
    merge_target_channel,
    place,
    post_process,
    stack,
)
from astroflow.pipeline import pipeline


@pipeline("Default")
def merge_mosaic(mode, mosaic, config):
    """Build a complete mosaic, end to end.

    Strategy: for each `(target, filter)` channel, stack each
    constituent run and post-process each master, then HDR-combine the
    per-stack post-processed masters into one channel master. For each
    filter, place every panel into the mosaic viewbox and pairwise-merge
    inside each strip, then merge rows. Returns `{filter: sink_metadata}`.

    Every @makes_image op call auto-chains a `compute_image_metadata`
    node behind the scenes (handled in `cache/cached_decorator.py`), so
    each variable name here holds a metadata handle rather than the
    raw file path. The `_after=` kwarg carries ordering edges for ops
    whose signature doesn't include the upstream master as a parameter
    (`post_process_stack(stack, mock)`, `merge_target_channel(...)`
    look up their inputs from disk).
    """
    mock = config.development_settings.mock_pixinsight
    impl = config.implementation

    # Resolve TargetHash -> Target once so the per-strip placement can
    # look up Target objects by id without re-iterating mosaic.targets.
    targets_by_id = {t.id: t for t in mosaic.targets}

    # ------------------------------------------------------------------
    # Per-(target, filter) channel masters.
    # ------------------------------------------------------------------
    channels: dict = {}  # (target_id, filter) -> ImageMetadata handle
    for target in mosaic.targets:
        for filter in target.filters:
            stacks = target.stacks_for(filter)
            stack_pps = []
            for s in stacks:
                master = stack(s, config)
                stack_pps.append(post_process(s, mock, _after=(master,)))
            channels[(target.id, filter)] = merge_target_channel(
                target, filter, stacks, config, _after=tuple(stack_pps),
            )

    # ------------------------------------------------------------------
    # Per filter: place panels into the mosaic viewbox, pairwise-merge
    # inside each strip, then merge across strips. Missing panels are
    # acceptable — the trace just skips them and merges what's present.
    # ------------------------------------------------------------------
    out: dict = {}
    for filter in mosaic.filters:
        viewbox = mosaic.viewbox(mode)

        rows = []
        for strip in mosaic.strips:
            placed = [
                place(mode, mosaic, targets_by_id[tid], filter, viewbox,
                      channels[(tid, filter)])
                for tid in strip if (tid, filter) in channels
            ]
            if not placed:
                continue
            row = placed[0]
            for p in placed[1:]:
                row = merge_images(mode, mosaic, filter, impl, mock, row, p)
            rows.append(row)

        if not rows:
            continue
        merged = rows[0]
        for r in rows[1:]:
            merged = merge_images(mode, mosaic, filter, impl, mock, merged, r)
        out[filter] = merged

    return out
