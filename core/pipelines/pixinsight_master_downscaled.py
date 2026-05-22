"""PixInsight master pipeline at 8x downscaled resolution.

Same algorithm as `PixInsight Master`, but every input frame —
lights AND the calibration masters they reference — is block-mean
downscaled 8x BEFORE WBPP runs. The whole point of this pipeline
is to cut WBPP/plate-solving/registration time by ~64x, which is
only achievable if WBPP itself sees the small frames.

Structurally the only differences from `pixinsight_master.py` are
which upstream ops it composes:

  PixInsight Master:                  PixInsight Downscaled:
    calibrate(s, config)                calibrate_downscaled(s, config)
    stack_files(s, config)              stack_files_downscaled(s, config)
    post_process_stack(...)             post_process_stack_downscaled(...)
    merge_target_channel_master(...)    merge_target_channel_downscaled_master(...)
    store_mosaic_color_channel_master   store_mosaic_color_channel_downscaled_master

`place_image` and `merge_images` are identical in both — both ops
are pipeline-agnostic (viewbox / inputs carry the discriminator).
"""
from astroflow.pipeline import pipeline
from mosaics.final_storage import store_mosaic_color_channel_downscaled_master
from mosaics.merging.pixinsight import merge_images
from mosaics.placement.astroflow import place_image
from mosaics.types import Mosaic
from stacks.compute_graph import calibrate_downscaled
from stacks.composite_stack import Stack
from stacks.post_processing.pixinsight import post_process_stack_downscaled
from stacks.stacking.pixinsight import stack_files_downscaled
from targets.merging.pixinsight import merge_target_channel_downscaled_master
from targets.types import Target


def _post_processed_master(s, config, mock):
    """Calibrate (downscaled cal masters) → stack downscaled →
    post-process. The `stack_files_downscaled` op handles the input-
    frame downscale and the WBPP run."""
    cal = calibrate_downscaled(s, config)
    master = stack_files_downscaled(s, config, _after=(cal,))
    return post_process_stack_downscaled(s, mock, _after=(master,))


def _channels_for_target(target, config, mock):
    return {
        filter: merge_target_channel_downscaled_master(
            target, filter, target.stacks_for(filter), config,
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
            place_image(mosaic, targets_by_id[tid], filter, viewbox,
                        channels[(tid, filter)])
            for tid in strip if (tid, filter) in channels
        ]
        if not placed:
            return None
        return _pairwise_merge(placed, mosaic, filter, mock)

    out = {}
    for filter in mosaic.filters:
        viewbox = mosaic.viewbox("DownscaledMaster")
        rows = [r for r in (
            merged_strip(strip, filter, viewbox) for strip in mosaic.strips
        ) if r is not None]
        if rows:
            merged = _pairwise_merge(rows, mosaic, filter, mock)
            out[filter] = store_mosaic_color_channel_downscaled_master(
                mosaic, filter, mock, merged,
            )

    return out


def _pairwise_merge(items, mosaic, filter, mock):
    acc = items[0]
    for item in items[1:]:
        acc = merge_images(mosaic, filter, mock, acc, item)
    return acc
