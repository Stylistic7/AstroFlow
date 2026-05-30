"""Merge per-stack masters into one per-(target, filter) master (OSS path).

n=1: passthrough copy. n>1: astroalign-register all to the longest
exposure, then inverse-variance weighted mean (weight =
n_frames * exposure^2 — the shot-noise-dominated optimum). Saturation
is detected and surfaced as a concern; the OSS path's weighted mean
underestimates SNR over saturated cores compared to true HDR
composition (see PIXINSIGHT path for that case).
"""
import os
import shutil
from pathlib import Path

import astropy.io.fits as pyfits
import numpy as np
import structlog
from astroalign import register as aa_register
from beartype import beartype

import concerns
from decisions.settings import ProcessingConfig
from decorators import node
from file_io.generic import read_image
from file_io.types import FitsFilePath, WithSidecar
from products.identity import (
    PresentationType,
    TargetChannelId,
    StackId,
)
from stacks.composite_stack import Stack
from targets.target_channel import TargetChannel
from targets.merging.pixinsight import _merge_target_channel_narrow
from targets.types import Target

THUMBNAIL = PresentationType.THUMBNAIL
TILE_PYRAMID = PresentationType.TILE_PYRAMID

log = structlog.get_logger()

# Most stack masters are written as float32 in [0, 1] but XISF / FITS
# masters from heterogeneous sources can carry uint16 saturation at
# 65535. Treat anything ≥ this fraction of the per-image max as saturated.
_SATURATION_FRACTION = 0.999

@node(
    identity=TargetChannelId,
    cache_key_narrow=_merge_target_channel_narrow,
    auto_memory=True,
)
def merge_target_channel_oss(
    target: Target, filter: str, stacks: tuple[Stack, ...],
    config: ProcessingConfig,
    *masters,
) -> WithSidecar[FitsFilePath]:
    """Variance-weighted combine of per-stack masters into one
    (target, filter) channel (OSS path).

    `*masters` is variadic so the trace detects each master as a
    top-level Node dep (nested Nodes inside tuples aren't
    introspected). Pipeline-agnostic between full-res and downscaled."""
    assert len(stacks) >= 1, "merge_target_channel_oss needs at least one stack"
    output_filename = merge_target_channel_oss.filename_fn(
        target, filter, stacks, config,
    )
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    master_paths = [m.filename for m in masters]

    if len(master_paths) == 1:
        shutil.copy(master_paths[0], output_filename)
        return WithSidecar(FitsFilePath(output_filename))

    stack_master_pairs = sorted(
        zip(stacks, master_paths), key=lambda p: -float(p[0].exposure),
    )
    sorted_stacks, sorted_master_paths = zip(*stack_master_pairs)
    masters_data = [
        read_image(p).astype(np.float32, copy=False)
        for p in sorted_master_paths
    ]
    # NOTE: variable name reuse — `masters` was the data-dep arg above;
    # below it shadows with the loaded ndarrays. Keep them straight.
    masters = masters_data

    reference = masters[0]
    aligned = [reference]
    for i, m in enumerate(masters[1:], start=1):
        try:
            registered, _ = aa_register(m, reference)
            aligned.append(np.nan_to_num(registered, nan=0.0).astype(np.float32, copy=False))
        except Exception as e:
            log.warning(
                "Failed to register stack master to reference; dropping",
                stack=str(sorted_stacks[i]), error=str(e),
            )

    # Saturation check (any input over threshold) — concern-emit, not fatal.
    for m, s in zip(aligned, sorted_stacks):
        if np.any(m >= _SATURATION_FRACTION * np.max(m)):
            concerns.warn(
                "Saturated pixels in HDR input on OSS path",
                detail=(
                    f"Stack {s.stable_hash} contributed saturated pixels "
                    f"to the (target, filter) merge. The OSS path uses "
                    f"a weighted mean and will underestimate SNR over "
                    f"saturated regions; the PixInsight HDRComposition "
                    f"path handles this case correctly."
                ),
                stack_id=s.stable_hash,
                target_id=target.stable_hash,
                filter=filter,
            )
            break

    # Inverse-variance weighting under shot-noise-dominated regime:
    # SNR² scales with total signal = n_frames * exposure. Variance
    # contribution of each input scales with 1/exposure² × n_frames,
    # so the optimal weight is n_frames × exposure².
    weights = np.array(
        [float(s.exposure) ** 2 * len(s.all_frames()) for s in sorted_stacks],
        dtype=np.float64,
    )
    weights = weights[: len(aligned)]  # account for any registration drops
    merged = np.average(np.stack(aligned), axis=0, weights=weights).astype(np.float32)

    header = pyfits.Header()
    header["IMAGETYP"] = "TargetChannel Master"
    header["FILTER"] = filter
    header["STBLHASH"] = target.stable_hash
    header["NSOURCES"] = len(aligned)
    header["PIPELINE"] = "AstroFlow-OSS"
    pyfits.PrimaryHDU(data=merged, header=header).writeto(
        str(output_filename), overwrite=True,
    )

    log.info(
        "OSS target-channel merge complete",
        target=target.stable_hash, filter=filter, sources=len(aligned),
        output=str(output_filename),
    )
    return WithSidecar(FitsFilePath(output_filename))
