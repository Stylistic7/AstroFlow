"""Reference pipeline: full-resolution PixInsight master, end-to-end.

WBPP-style stacking, per-stack PixInsight post-processing, HDR channel
combine, photometric mosaic place + merge. Three entry points, one
per cart-entry kind:

* `for_stack(stack, config)`   → one post-processed stack master.
* `for_target(target, config)` → per-filter channel masters for one
                                 target.
* `for_mosaic(mosaic, config)` → per-filter mosaic outputs (the
                                 headline result).

All three share the per-stack chain (`_post_processed_master`); the
target entry composes that across a target's filters, and the mosaic
entry composes the target step across a mosaic's targets and adds
panel placement + pairwise merge.

To fork this pipeline (e.g. swap MSGC for GraXpert), copy this file
to `~/AstroFlow/pipelines/<YourName>.py`, change the `@pipeline("…")`
display name, and edit the call you want to replace. All three entry
points are independently customisable.
"""
from astroflow.pipeline import pipeline
# Explicit per-backend imports — reading this file tells you which
# implementation each step uses. An "OSS Master" sibling would import
# `stack_files` from `stacks.stacking.oss` etc.; the diff between
# the two files IS the algorithm change.
from mosaics.merging.pixinsight import merge_images
from mosaics.placement.astroflow import place_image
from mosaics.types import Mosaic
from previews.types import Implementation, QualityLevel
from stacks.compute_graph import calibrate
from stacks.composite_stack import Stack
from stacks.post_processing.pixinsight import post_process_stack
from stacks.stacking.pixinsight import stack_files
from targets.merging.pixinsight import merge_target_channel
from targets.types import Target


# This pipeline's output identity. Used internally by the ops that
# still branch on QualityLevel (place_image, merge_images, etc.);
# Phase B-3 splits those into per-quality ops and removes the constant.
_MODE = QualityLevel.MASTER
_IMPL = Implementation.PIXINSIGHT


def _post_processed_master(s, config, mock):
    """Calibrate → stack → post-process one stack. Returns the
    post-processed master's metadata handle. The `_after=` ordering
    carries disk-side data flow: `post_process_stack` looks up its
    input via filename_fn, so an ordering edge to the master is
    enough — no value flow."""
    cal = calibrate(s, _MODE, config)
    master = stack_files(s, _MODE, config, _after=(cal,))
    return post_process_stack(s, mock, _after=(master,))


def _channels_for_target(target, config, mock):
    """Per-filter HDR-combined channel masters for one target."""
    return {
        filter: merge_target_channel(
            target, filter, target.stacks_for(filter), _MODE, config,
            _after=tuple(
                _post_processed_master(s, config, mock)
                for s in target.stacks_for(filter)
            ),
        )
        for filter in target.filters
    }


@pipeline("PixInsight Master", processes=Stack)
def for_stack(stack, config):
    """One post-processed PixInsight master for one stack."""
    return _post_processed_master(
        stack, config, config.development_settings.mock_pixinsight,
    )


@pipeline("PixInsight Master", processes=Target)
def for_target(target, config):
    """Per-filter channel masters for one target."""
    return _channels_for_target(
        target, config, config.development_settings.mock_pixinsight,
    )


@pipeline("PixInsight Master", processes=Mosaic)
def for_mosaic(mosaic, config):
    """Build a complete mosaic, end to end. Returns
    `{filter: sink_metadata}`.

    Per (target, filter), run the shared `_channels_for_target` step.
    Per filter, place every panel into the mosaic viewbox and
    pairwise-merge inside each strip, then merge rows. Missing panels
    are skipped — the trace just merges what's present.
    """
    mock = config.development_settings.mock_pixinsight
    targets_by_id = {t.id: t for t in mosaic.targets}

    # Per-(target, filter) channel masters — same chain as the
    # target-only entry, flattened across the mosaic's targets.
    channels = {
        (target.id, filter): sink
        for target in mosaic.targets
        for filter, sink in _channels_for_target(target, config, mock).items()
    }

    def merged_strip(strip, filter, viewbox):
        placed = [
            place_image(_MODE, mosaic, targets_by_id[tid], filter, viewbox,
                        channels[(tid, filter)])
            for tid in strip if (tid, filter) in channels
        ]
        if not placed:
            return None
        return _pairwise_merge(placed, mosaic, filter, mock)

    out = {}
    for filter in mosaic.filters:
        viewbox = mosaic.viewbox(_MODE)
        rows = [r for r in (
            merged_strip(strip, filter, viewbox) for strip in mosaic.strips
        ) if r is not None]
        if rows:
            out[filter] = _pairwise_merge(rows, mosaic, filter, mock)

    return out


def _pairwise_merge(items, mosaic, filter, mock):
    """Left-fold merge_images over a list of >=1 placed/merged
    metadata handles. Returns the final sink Node."""
    acc = items[0]
    for item in items[1:]:
        acc = merge_images(_MODE, mosaic, filter, _IMPL, mock, acc, item)
    return acc
