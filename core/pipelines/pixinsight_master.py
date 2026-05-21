"""Reference pipeline: full-resolution PixInsight master, end-to-end.

The canonical pipeline (replaces the legacy `Default` name): WBPP-style
stacking, per-stack PixInsight post-processing, HDR channel combine,
photometric mosaic place + merge. Opt in by selecting `"PixInsight
Master"` from the pipeline dropdown when adding a mosaic to the
shopping cart. Cart entry shape: `("mosaic", mosaic_id, "PixInsight
Master")`.

Bundled pipeline names (`"PixInsight Master"`, `"PixInsight Master
Downscaled"`, `"Subframe"`, `"OSS Master"`) and registered user
pipelines (in `~/AstroFlow/pipelines/`) live in the same dropdown. The
choice is per-cart-entry, so the same mosaic can be added twice with
two pipelines for side-by-side comparison. Pipeline names never reach
any @node op's arg hash — ops are pipeline-agnostic and share cache
across pipelines that call them with identical args.

To create a variant that swaps a step (e.g. MSGC → GraXpert), copy
this file to `~/AstroFlow/pipelines/<YourName>.py`, change the
`@pipeline("…")` display name, and edit the call you want to
replace.
"""
from astroflow.pipeline import pipeline
# Explicit per-backend imports — reading the pipeline file tells you
# exactly which implementation each step uses. An "OSS Master" sibling
# pipeline would import `stack_files` from `stacks.stacking.oss` etc.,
# and the diff between the two files is the algorithm change.
from mosaics.merging.pixinsight import merge_images
from mosaics.placement.astroflow import place_image
from stacks.compute_graph import calibrate
from stacks.post_processing.pixinsight import post_process_stack
from stacks.stacking.pixinsight import stack_files
from targets.merging.pixinsight import merge_target_channel


@pipeline("PixInsight Master")
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
                # Calibration deps are stacked first. The Node returned
                # by `calibrate` expands to a sub-graph at runtime
                # (the per-(bias, dark, flat) stack tasks); `_after=(cal,)`
                # on the light stack waits on that sub-graph's
                # `calibration_done` sentinel.
                cal = calibrate(s, config)
                master = stack_files(s, config, _after=(cal,))
                stack_pps.append(
                    post_process_stack(s, mock, _after=(master,))
                )
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
                place_image(mode, mosaic, targets_by_id[tid], filter, viewbox,
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
