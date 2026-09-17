"""Hybrid pipeline at 8× downscaled resolution: PixInsight stacking +
OSS mosaic merging.

Same backbone as `pixinsight_master_downscaled.py` — every light frame
is block-mean downscaled 8× BEFORE PixInsight's WBPP runs (the
calibration masters are built at full resolution and downscaled at the
boundary), then per-stack post-processing (WriteAstrometricSolution +
SPFC + MGC) and HDR per-target combine all happen at the downscaled
resolution. The mosaic merging step then uses
`merge_images_oss` (distance-weighted feathering + per-overlap flux
equalization) in place of PI's PhotometricMosaic, to eliminate the
panel transitions visible on large multi-panel mosaics.

The 8× downscale cuts WBPP / per-stack PI time ~64× — recommended for
fast iteration on the panel-blending behavior before committing a
full-resolution run.

Three entry points, matching the PI Downscaled pipeline.
"""
from astroflow.pipeline import pipeline
from dependency_graph.tracing import workflow_group
from mosaics.types import Mosaic
import pipelines.hybrid_pi_oss_mosaic as hybrid
from stacks.calibration_selection import calibrate
from stacks.composite_stack import Stack
from previews.types import DOWNSCALE_FACTORS, DOWNSCALED_MASTER
from stacks.downscale import (
    DownscaledStack, downscale_dest_dir, downscale_master, downscale_stack,
)
from stacks.post_processing.pixinsight import post_process_stack
from stacks.stacking.pixinsight import stack_files
from targets.types import Target

FACTOR = DOWNSCALE_FACTORS[DOWNSCALED_MASTER]


@workflow_group
def post_processed_master(stack, config):
    """The full-resolution recipe with the downscale calls added: the light
    frames are seen through a `DownscaledStack` (written by
    `downscale_stack`), the calibration masters are built at full resolution
    exactly as `PixInsight Master` builds them and downscaled at the
    boundary. Shared across the stack / target / mosaic entry points."""
    ds = DownscaledStack.of(stack, FACTOR, downscale_dest_dir())
    frames = downscale_stack(ds)
    cal_masters = [downscale_master(FACTOR, m) for m in calibrate(stack, config)]
    master = stack_files(ds, config, *cal_masters, _after=(frames,))
    return post_process_stack(ds, config, master)


@pipeline("Hybrid: PI Downscaled + OSS Mosaic", processes=Stack)
def for_stack(stack, config):
    """One stack's post-processed 8×-downscaled master."""
    return post_processed_master(stack, config)


@pipeline("Hybrid: PI Downscaled + OSS Mosaic", processes=Target)
def for_target(target, config):
    """Per-filter 8×-downscaled channel masters for one target."""
    return hybrid.channels_for_target(target, config, post_processed_master)


@pipeline("Hybrid: PI Downscaled + OSS Mosaic", processes=Mosaic)
def for_mosaic(mosaic, config):
    """Build an 8×-downscaled mosaic with PI-stacked panels + OSS
    panel-blending, then the colour deliverables — the full-resolution
    hybrid's mosaic stage, unchanged."""
    return hybrid.mosaic_stage(mosaic, config, post_processed_master)
