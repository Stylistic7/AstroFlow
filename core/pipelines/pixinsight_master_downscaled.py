"""PixInsight master pipeline at 8x downscaled resolution.

Same algorithm as `PixInsight Master`, but every input frame —
lights AND the calibration masters they reference — is block-mean
downscaled 8x BEFORE WBPP runs. The whole point of this pipeline
is to cut WBPP/plate-solving/registration time by ~64x, which is
only achievable if WBPP itself sees the small frames.

Structurally the only differences from `pixinsight_master.py` are
which upstream stack-side ops it composes:

  PixInsight Master:                    PixInsight Downscaled:
    calibrate(...)                        calibrate_downscaled(...)
    stack_files(...)                      stack_files_downscaled(...)
    post_process_stack(...)               post_process_stack_downscaled(...)

`merge_target_channel`, `place_image`, `merge_images`, and
`store_mosaic_color_channel_xisf` are SHARED — the data deps
(different per-stack masters, different viewbox) keep the outputs
in distinct on-disk files via the call-hash filename.
"""
from astroflow.pipeline import pipeline
from mosaics.final_storage import store_mosaic_color_channel_downscaled_master
from mosaics.merging.pixinsight import merge_images_downscaled_master
from mosaics.placement.astroflow import place_image
from mosaics.types import Mosaic
from stacks.compute_graph import calibrate_downscaled
from stacks.composite_stack import Stack
from stacks.post_processing.pixinsight import post_process_stack_downscaled
from stacks.stacking.pixinsight import stack_files_downscaled
from targets.merging.pixinsight import merge_target_channel_downscaled_master
from targets.types import Target


def _post_processed_master(s, config, mock):
    cal = calibrate_downscaled(s, config)
    master = stack_files_downscaled(s, config, _after=(cal,))
    return post_process_stack_downscaled(s, mock, master)


def _channels_for_target(target, config, mock):
    return {
        filter: merge_target_channel_downscaled_master(
            target, filter, target.stacks_for(filter), config,
            *[_post_processed_master(s, config, mock)
              for s in target.stacks_for(filter)],
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

    out = {}
    for filter in mosaic.filters:
        viewbox = mosaic.viewbox("DownscaledMaster")
        placed = [
            place_image(mosaic, targets_by_id[tid], filter, viewbox,
                        channels[(tid, filter)])
            for strip in mosaic.strips
            for tid in strip if (tid, filter) in channels
        ]
        if not placed:
            continue
        merged = placed[0] if len(placed) == 1 else merge_images_downscaled_master(
            mosaic, filter, mock, *placed,
        )
        out[filter] = store_mosaic_color_channel_downscaled_master(
            mosaic, filter, mock, merged,
        )

    return out
