"""PixInsight master at 8x downscaled resolution.

Same algorithm as `PixInsight Master`, with one extra step inserted
after stacking: `downscale_master` block-mean-downscales the
full-resolution master to 8x. Gradient correction
(`post_process_stack_downscaled`) then runs on the small image.

Critical: `stack_files` is invoked with the SAME args as in the
`PixInsight Master` pipeline, so the diskcache entry is shared
between the two pipelines per (stack, config). Adding this mosaic to
the cart with both pipelines runs `stack_files` exactly once and
re-uses its master for both. The downscale step and everything
downstream of it has its own cache entries, sized to the downscaled
work.

The only structural difference from `pixinsight_master.py`:

  PixInsight Master:                PixInsight Downscaled:
    stack_files(s, config)            stack_files(s, config)
                                      downscale_master(s, config)
    post_process_stack(...)           post_process_stack_downscaled(...)
"""
from astroflow.pipeline import pipeline
from mosaics.final_storage import store_mosaic_color_channel_xisf
from mosaics.merging.pixinsight import merge_images
from mosaics.placement.astroflow import place_image
from mosaics.types import Mosaic
from previews.types import PIXINSIGHT, QualityLevel
from stacks.compute_graph import calibrate
from stacks.composite_stack import Stack
from stacks.downscale_master import downscale_master
from stacks.post_processing.pixinsight import post_process_stack_downscaled
from stacks.stacking.pixinsight import stack_files
from targets.merging.pixinsight import merge_target_channel
from targets.types import Target


_MODE = QualityLevel.DOWNSCALED_MASTER
_IMPL = PIXINSIGHT


def _post_processed_master(s, config, mock):
    """Calibrate → stack (full-res) → downscale → post-process the
    downscaled image. The `stack_files` call is byte-identical to the
    one in `PixInsight Master` — they share the diskcache entry."""
    cal = calibrate(s, config)
    master = stack_files(s, config, _after=(cal,))
    downscaled = downscale_master(s, config, _after=(master,))
    return post_process_stack_downscaled(s, mock, _after=(downscaled,))


def _channels_for_target(target, config, mock):
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


@pipeline("PixInsight Downscaled", processes=Stack)
def for_stack(stack, config):
    return _post_processed_master(
        stack, config, config.development_settings.mock_pixinsight,
    )


@pipeline("PixInsight Downscaled", processes=Target)
def for_target(target, config):
    return _channels_for_target(
        target, config, config.development_settings.mock_pixinsight,
    )


@pipeline("PixInsight Downscaled", processes=Mosaic)
def for_mosaic(mosaic, config):
    mock = config.development_settings.mock_pixinsight
    targets_by_id = {t.id: t for t in mosaic.targets}

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
            merged = _pairwise_merge(rows, mosaic, filter, mock)
            out[filter] = store_mosaic_color_channel_xisf(
                _MODE, mosaic, filter, mock, merged,
            )

    return out


def _pairwise_merge(items, mosaic, filter, mock):
    acc = items[0]
    for item in items[1:]:
        acc = merge_images(_MODE, mosaic, filter, _IMPL, mock, acc, item)
    return acc
