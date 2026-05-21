"""PixInsight master at 8x downscaled resolution.

Same algorithm as `PixInsight Master`, but the stacking step produces
a downscaled master (via PixInsight's RegistrationScale parameter) and
gradient correction runs on the small image. Cheap to compute, useful
as a fast end-to-end check before launching the full-res run.

The only difference from `pixinsight_master.py` is the post-processing
op: `post_process_stack_downscaled` instead of `post_process_stack`.
Everything else — calibration, stacking, channel HDR-combine, panel
placement, mosaic merge — is the same call shape. The legacy
`build_mosaic_subgraph(mode=DOWNSCALED_MASTER)` delegation is gone.

Cache sharing with `PixInsight Master`: today the `stack_files` cache
key includes `config.quality`, so this pipeline's stack_files calls
don't hit cache entries produced by the master pipeline. Cross-
pipeline `stack_files` sharing would require either dropping
`config.quality` from the stack_files signature (and routing the
downscale via a separate op composed on top) or splitting stack_files
into two distinct ops. Deferred — see plan B-3 / B-5.
"""
from astroflow.pipeline import pipeline
from mosaics.merging.pixinsight import merge_images
from mosaics.placement.astroflow import place_image
from stacks.compute_graph import calibrate
from stacks.post_processing.pixinsight import post_process_stack_downscaled
from stacks.stacking.pixinsight import stack_files
from targets.merging.pixinsight import merge_target_channel


@pipeline("PixInsight Master Downscaled")
def merge_mosaic(mode, mosaic, config):
    """Build a downscaled mosaic, end to end.

    Structurally identical to `pixinsight_master.merge_mosaic` except
    each stack's gradient correction goes through
    `post_process_stack_downscaled` (which operates on the downscaled
    master rather than the full-resolution one).

    `mode` is `DOWNSCALED_MASTER` per the cart dispatcher's
    `_pipeline_to_mode_impl` mapping; `config.quality` is
    `DOWNSCALED_MASTER` for the same reason. The pipeline-name choice
    propagates through both.
    """
    mock = config.development_settings.mock_pixinsight
    impl = config.implementation

    targets_by_id = {t.id: t for t in mosaic.targets}

    channels: dict = {}
    for target in mosaic.targets:
        for filter in target.filters:
            stacks = target.stacks_for(filter)
            stack_pps = []
            for s in stacks:
                cal = calibrate(s, config)
                master = stack_files(s, config, _after=(cal,))
                stack_pps.append(
                    post_process_stack_downscaled(s, mock, _after=(master,))
                )
            channels[(target.id, filter)] = merge_target_channel(
                target, filter, stacks, config, _after=tuple(stack_pps),
            )

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
