"""Reference pipeline: full-resolution PixInsight master, end-to-end.

The canonical pipeline (replaces the legacy `Default` name): WBPP-style
stacking, per-stack PixInsight post-processing, HDR channel combine,
photometric mosaic place + merge.

Three entry points, one per cart-entry kind:

* `for_stack(mode, stack, config)`   → produces a single post-processed
                                       stack master.
* `for_target(mode, target, config)` → produces per-filter channel
                                       masters for one target.
* `for_mosaic(mode, mosaic, config)` → produces per-filter mosaic
                                       outputs (the headline result).

All three share the per-stack chain (`_for_stack_internal`); the
target entry composes that across a target's filters, and the mosaic
entry composes the target step across a mosaic's targets and adds
panel placement + pairwise merge.

To fork this pipeline (e.g. swap MSGC for GraXpert), copy this file
to `~/AstroFlow/pipelines/<YourName>.py`, change the `@pipeline("…")`
display name, and edit the call you want to replace. All three
entry points are independently customisable.
"""
from astroflow.pipeline import pipeline
# Explicit per-backend imports — reading the pipeline file tells you
# exactly which implementation each step uses. An "OSS Master" sibling
# would import `stack_files` from `stacks.stacking.oss`, etc.; the diff
# between the two files is the algorithm change.
from mosaics.merging.pixinsight import merge_images
from mosaics.placement.astroflow import place_image
from mosaics.types import Mosaic
from stacks.compute_graph import calibrate
from stacks.composite_stack import Stack
from stacks.post_processing.pixinsight import post_process_stack
from stacks.stacking.pixinsight import stack_files
from targets.merging.pixinsight import merge_target_channel
from targets.types import Target


def _post_processed_master(s, config, mock):
    """Calibrate → stack → post-process one stack. Returns the
    post-processed master's metadata handle. Shared between the
    stack, target, and mosaic entry points so each step is described
    once. The `_after=` ordering carries the disk-side data flow:
    `post_process_stack` looks up its input via filename_fn, so we
    only need an ordering edge to the master, not a value flow."""
    cal = calibrate(s, config)
    master = stack_files(s, config, _after=(cal,))
    return post_process_stack(s, mock, _after=(master,))


def _channels_for_target(target, config, mock):
    """Per-filter HDR-combined channel masters for one target. The
    Node handles in `stack_pps` carry their post-processed metadata;
    `merge_target_channel` looks up the per-stack files via
    filename_fn, so the `_after=` carries the ordering edge."""
    channels: dict = {}
    for filter in target.filters:
        stacks = target.stacks_for(filter)
        stack_pps = [_post_processed_master(s, config, mock) for s in stacks]
        channels[filter] = merge_target_channel(
            target, filter, stacks, config, _after=tuple(stack_pps),
        )
    return channels


@pipeline("PixInsight Master", processes=Stack)
def for_stack(mode, stack, config):
    """Produce one post-processed PixInsight master for one stack."""
    del mode  # implicit in the pipeline's identity
    mock = config.development_settings.mock_pixinsight
    return _post_processed_master(stack, config, mock)


@pipeline("PixInsight Master", processes=Target)
def for_target(mode, target, config):
    """Per-filter channel masters for one target."""
    del mode
    mock = config.development_settings.mock_pixinsight
    return _channels_for_target(target, config, mock)


@pipeline("PixInsight Master", processes=Mosaic)
def for_mosaic(mode, mosaic, config):
    """Build a complete mosaic, end to end. Returns
    `{filter: sink_metadata}`.

    Strategy: per (target, filter) channel, run the shared
    `_channels_for_target` step. Per filter, place every panel into
    the mosaic viewbox and pairwise-merge inside each strip, then
    merge rows.
    """
    mock = config.development_settings.mock_pixinsight
    impl = config.implementation

    targets_by_id = {t.id: t for t in mosaic.targets}

    # Per-(target, filter) channel masters — same chain as the
    # target-only entry, just flattened across all of the mosaic's
    # targets.
    channels: dict = {}
    for target in mosaic.targets:
        for filter, sink in _channels_for_target(target, config, mock).items():
            channels[(target.id, filter)] = sink

    # Per filter: place panels into the mosaic viewbox, pairwise-merge
    # inside each strip, then merge across strips. Missing panels are
    # acceptable — the trace just skips them and merges what's present.
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
