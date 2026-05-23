"""PixInsight master pipeline at 8x downscaled resolution.

Same algorithm as `PixInsight Master`, but every input frame —
lights AND the calibration masters they reference — is block-mean
downscaled 8x BEFORE WBPP runs. The whole point of this pipeline
is to cut WBPP/plate-solving/registration time by ~64x, which is
only achievable if WBPP itself sees the small frames.

Structurally identical to `pixinsight_master.py`: it imports the
SAME `post_processed_master` shared core and passes a
non-identity `downscaler`. Every op downstream — `stack_files`,
`post_process_stack`, `merge_target_channel`, `merge_images`,
`store_mosaic_color_channel_xisf` — is the same op. The
framework distinguishes the two pipelines' outputs via the
call_hash: each op's bound args include its upstream data deps,
whose content_hashes differ between the two pipelines, so the
on-disk filenames differ.
"""
from astroflow.pipeline import pipeline
from mosaics.final_storage import store_mosaic_color_channel_xisf
from mosaics.merging.pixinsight import merge_images
from mosaics.placement.astroflow import place_image
from mosaics.types import Mosaic
from stacks.composite_stack import Stack
from stacks.downscale import downscale_stack
from stacks.pipelines import post_processed_master
from targets.merging.pixinsight import merge_target_channel
from targets.types import Target


def _post_processed_master(s, config, mock):
    return post_processed_master(
        s, config, mock,
        downscaler=lambda x: downscale_stack(x, 8),
    )


def _channels_for_target(target, config, mock):
    return {
        filter: merge_target_channel(
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
        merged = placed[0] if len(placed) == 1 else merge_images(
            mosaic, filter, mock, *placed,
        )
        out[filter] = store_mosaic_color_channel_xisf(
            mosaic, filter, mock, merged,
        )

    return out
