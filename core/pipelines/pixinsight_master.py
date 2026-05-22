"""Reference pipeline: full-resolution PixInsight master, end-to-end.

WBPP-style stacking, per-stack PixInsight post-processing, HDR channel
combine, photometric mosaic place + merge. Three entry points:

* `for_stack(stack, config)`   → one post-processed stack master.
* `for_target(target, config)` → per-filter channel masters.
* `for_mosaic(mosaic, config)` → per-filter mosaic outputs (headline).

All ops compose by data flow — each consumer receives its upstream's
`WithSidecar` handle as a positional arg. No `filename_fn` lookups
inside op bodies; no mode/level threading. Different pipelines that
share the same op + same args share the cache; otherwise the per-
op call-hash filename keeps outputs from colliding.

To fork this pipeline (e.g. swap MSGC for GraXpert), copy this file
to `~/AstroFlow/pipelines/<YourName>.py`, change the `@pipeline("…")`
display name, and edit the call you want to replace.
"""
from astroflow.pipeline import pipeline
from mosaics.final_storage import store_mosaic_color_channel_xisf
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
    post-processed master's `WithSidecar` handle. The master flows
    into `post_process_stack` as a data dep (positional arg) —
    no filename lookup."""
    cal = calibrate(s, config)
    master = stack_files(s, config, _after=(cal,))
    return post_process_stack(s, mock, master)


def _channels_for_target(target, config, mock):
    """Per-filter HDR-combined channel masters. The `masters` tuple
    is the last positional arg to merge_target_channel — Node args
    in trace context must be positional (the trace machinery doesn't
    accept kwargs)."""
    return {
        filter: merge_target_channel(
            target, filter, target.stacks_for(filter), config,
            tuple(
                _post_processed_master(s, config, mock)
                for s in target.stacks_for(filter)
            ),
        )
        for filter in target.filters
    }


@pipeline("PixInsight Master", processes=Stack)
def for_stack(stack, config):
    return _post_processed_master(
        stack, config, config.development_settings.mock_pixinsight,
    )


@pipeline("PixInsight Master", processes=Target)
def for_target(target, config):
    return _channels_for_target(
        target, config, config.development_settings.mock_pixinsight,
    )


@pipeline("PixInsight Master", processes=Mosaic)
def for_mosaic(mosaic, config):
    """Build a complete mosaic. Returns `{filter: sink_metadata}`.
    Missing panels are skipped — the trace merges what's present."""
    mock = config.development_settings.mock_pixinsight
    targets_by_id = {t.id: t for t in mosaic.targets}

    channels = {
        (target.id, filter): sink
        for target in mosaic.targets
        for filter, sink in _channels_for_target(target, config, mock).items()
    }

    def merged_strip(strip, filter, viewbox):
        placed = [
            place_image(mosaic, targets_by_id[tid], filter, viewbox,
                        channels[(tid, filter)])
            for tid in strip if (tid, filter) in channels
        ]
        if not placed:
            return None
        return _pairwise_merge(placed, mosaic, filter, mock)

    out = {}
    for filter in mosaic.filters:
        viewbox = mosaic.viewbox("Master")
        rows = [r for r in (
            merged_strip(strip, filter, viewbox) for strip in mosaic.strips
        ) if r is not None]
        if rows:
            merged = _pairwise_merge(rows, mosaic, filter, mock)
            out[filter] = store_mosaic_color_channel_xisf(
                mosaic, filter, mock, merged,
            )

    return out


def _pairwise_merge(items, mosaic, filter, mock):
    acc = items[0]
    for item in items[1:]:
        acc = merge_images(mosaic, filter, mock, acc, item)
    return acc
