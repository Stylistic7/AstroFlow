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
from stacks.composite_stack import Stack
from stacks.pipelines import post_processed_master
from targets.merging.pixinsight import merge_target_channel
from targets.types import Target


def _post_processed_master(s, config, mock):
    """PI Master uses the unified stacking core with the IDENTITY
    downscaler (no downscaling — frames go to WBPP at full res)."""
    return post_processed_master(s, config, mock, downscaler=lambda x: x)


def _channels_for_target(target, config, mock):
    """Per-filter HDR-combined channel masters. Spreads the per-stack
    masters as variadic args (`*masters`) so each one shows up as a
    top-level Node dep at trace time — the trace doesn't introspect
    nested tuples for Node handles."""
    return {
        filter: merge_target_channel(
            target, filter, target.stacks_for(filter), config,
            *[_post_processed_master(s, config, mock)
              for s in target.stacks_for(filter)],
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
    Missing panels are skipped — the trace merges what's present.

    Per filter: place every panel into the mosaic viewbox, then
    spread all placed handles into ONE `merge_images` call. The
    op's body does the pairwise PhotometricMosaic tree inside a
    single Ray task — necessary because every pairwise call would
    share `(mosaic, filter, mock)` literals and collide on the
    call-hash filename otherwise."""
    mock = config.development_settings.mock_pixinsight
    targets_by_id = {t.id: t for t in mosaic.targets}

    channels = {
        (target.id, filter): sink
        for target in mosaic.targets
        for filter, sink in _channels_for_target(target, config, mock).items()
    }

    out = {}
    for filter in mosaic.filters:
        viewbox = mosaic.viewbox("Master")
        placed = [
            place_image(mosaic, targets_by_id[tid], filter, viewbox,
                        channels[(tid, filter)])
            for strip in mosaic.strips
            for tid in strip if (tid, filter) in channels
        ]
        if not placed:
            continue
        # Single panel: skip merge_images. The placed image IS the
        # whole-mosaic output for this filter; the store handles it
        # directly. This keeps single-panel mosaics from spending a
        # Ray task on a trivial copy.
        merged = placed[0] if len(placed) == 1 else merge_images(
            mosaic, filter, mock, *placed,
        )
        out[filter] = store_mosaic_color_channel_xisf(
            mosaic, filter, mock, merged,
        )

    return out
