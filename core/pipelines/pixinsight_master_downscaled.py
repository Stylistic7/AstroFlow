"""PixInsight master at 8x downscaled resolution.

Same algorithm as `PixInsight Master`, but stacking produces a
downscaled master (via PixInsight's RegistrationScale parameter) and
gradient correction runs on the small image. Cheap end-to-end check
before launching the full-res run.

The only diff from `pixinsight_master.py` is the per-stack chain's
post-processing op: `post_process_stack_downscaled` in place of
`post_process_stack`, and the `_MODE` constant.

Cache sharing with `PixInsight Master`: today `stack_files` takes
`mode: QualityLevel` as an explicit arg (it produces MASTER or
DOWNSCALED_MASTER), so this pipeline's stack_files calls don't hit
cache entries produced by the master pipeline. Cross-pipeline
`stack_files` sharing would require splitting stack_files into two
ops (one per quality) and composing a separate `downscale` op
above the master variant. Deferred — see plan B-3c.
"""
from astroflow.pipeline import pipeline
from mosaics.merging.pixinsight import merge_images
from mosaics.placement.astroflow import place_image
from mosaics.types import Mosaic
from previews.types import Implementation, QualityLevel
from stacks.compute_graph import calibrate
from stacks.composite_stack import Stack
from stacks.post_processing.pixinsight import post_process_stack_downscaled
from stacks.stacking.pixinsight import stack_files
from targets.merging.pixinsight import merge_target_channel
from targets.types import Target


_MODE = QualityLevel.DOWNSCALED_MASTER
_IMPL = Implementation.PIXINSIGHT


def _post_processed_master(s, config, mock):
    cal = calibrate(s, _MODE, config)
    master = stack_files(s, _MODE, config, _after=(cal,))
    return post_process_stack_downscaled(s, mock, _after=(master,))


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


@pipeline("PixInsight Master Downscaled", processes=Stack)
def for_stack(stack, config):
    return _post_processed_master(
        stack, config, config.development_settings.mock_pixinsight,
    )


@pipeline("PixInsight Master Downscaled", processes=Target)
def for_target(target, config):
    return _channels_for_target(
        target, config, config.development_settings.mock_pixinsight,
    )


@pipeline("PixInsight Master Downscaled", processes=Mosaic)
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
            out[filter] = _pairwise_merge(rows, mosaic, filter, mock)

    return out


def _pairwise_merge(items, mosaic, filter, mock):
    acc = items[0]
    for item in items[1:]:
        acc = merge_images(_MODE, mosaic, filter, _IMPL, mock, acc, item)
    return acc
