"""Reference pipeline: PixInsight defaults, end-to-end.

Demonstrates the whole-pipeline-as-plain-Python pattern (see the plan
at `~/.claude/plans/splendid-munching-rainbow.md`). Opt in via
`DevelopmentSettings(pipeline_name="Default")`; the `"mosaic"` cart
branch then dispatches to `merge_mosaic` below instead of the legacy
`build_mosaic_subgraph` chain.

This reference reuses the existing PixInsight @node ops verbatim — same
function objects, same cache keys. To create a variant that swaps a
step (e.g. MSGC → GraXpert), copy this file to
`~/AstroFlow/pipelines/<YourName>.py`, change the `@pipeline("…")`
display name, and edit the call you want to replace. A future
iteration of this layer will add the inline-`@stage` recipe pattern
that lets a single file fully express its own post-processing recipe.
"""
from astroflow.ops import (
    compute_image_metadata,
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

    Data flow handles most dependencies via `compute_image_metadata`
    threading; the few ops whose signatures don't carry their data dep
    (e.g. `post_process_stack(stack, mock)`, `merge_target_channel(...)`
    look up files internally) use the `_after=` kwarg to express
    ordering-only edges to the trace.
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

            # Per-stack: stack the lights, then post-process. Both ops
            # look up their input files internally; threading metadata
            # establishes data deps for `compute_image_metadata`, and
            # `_after=` carries the ordering edge to `post_process`.
            stack_pps = []
            for s in stacks:
                master = stack(s, config)
                master_meta = compute_image_metadata(master)
                pp = post_process(s, mock, _after=(master_meta,))
                stack_pps.append(compute_image_metadata(pp))

            # HDR-combine the per-stack post-processed masters into one
            # (target, filter) channel master. `merge_target_channel`
            # doesn't take the per-stack outputs as args (it resolves
            # them from the `stacks` tuple internally), so ordering
            # rides on `_after=`.
            channel = merge_target_channel(
                target, filter, stacks, config, _after=tuple(stack_pps),
            )
            channels[(target.id, filter)] = compute_image_metadata(channel)

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
            placed = []
            for tid in strip:
                if (tid, filter) not in channels:
                    continue  # missing panel in this filter
                target = targets_by_id[tid]
                p = place(
                    mode, mosaic, target, filter, viewbox,
                    channels[(tid, filter)],
                )
                placed.append(compute_image_metadata(p))
            if not placed:
                continue
            row = placed[0]
            for p in placed[1:]:
                row = compute_image_metadata(
                    merge_images(mode, mosaic, filter, impl, mock, row, p),
                )
            rows.append(row)

        if not rows:
            continue
        merged = rows[0]
        for r in rows[1:]:
            merged = compute_image_metadata(
                merge_images(mode, mosaic, filter, impl, mock, merged, r),
            )
        out[filter] = merged

    return out
