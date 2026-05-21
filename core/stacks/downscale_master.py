"""Downscale a stack master to produce a DOWNSCALED_MASTER artifact.

Separates the downscaling step from `stack_files`, which now always
produces a full-resolution master regardless of pipeline. The
`PixInsight Downscaled` pipeline composes `downscale_master` on top
of `stack_files`, so both PI Master and PI Downscaled pipelines
share the upstream `stack_files` diskcache entry per (stack, config)
call — the cache-sharing promise of the pipeline architecture.

Downstream ops that look up DOWNSCALED_MASTER via the registry
(`post_process_stack_downscaled`, `merge_target_channel` with
mode=DOWNSCALED_MASTER) automatically pick up this op's output path,
because @makes_image registers `downscale_master` as the producer
for that (identity, level) slot.
"""
import os
import numpy as np
import structlog

from decisions.settings import ProcessingConfig
from decorators import node
from file_io.generic import read_image
from file_io.types import XisfFilePath, WithSidecar
from file_io.xisf_file_io import write_xisf_image
from previews.types import (
    PIXINSIGHT, MASTER, DOWNSCALED_MASTER, DOWNSCALE_FACTORS,
)
from products import makes_image, get_filename_fn, MediaType
from products.identity import (
    StackId, CalibrationStackId, PresentationType, identity_for_stack,
)
from stacks.composite_stack import BaseStack

THUMBNAIL = PresentationType.THUMBNAIL
TILE_PYRAMID = PresentationType.TILE_PYRAMID

log = structlog.get_logger()


@node(auto_memory=True)
@makes_image(PIXINSIGHT, [DOWNSCALED_MASTER],
             identity=CalibrationStackId, presentations=[THUMBNAIL, TILE_PYRAMID],
             file_type=XisfFilePath)
@makes_image(PIXINSIGHT, [DOWNSCALED_MASTER],
             identity=StackId, presentations=[THUMBNAIL, TILE_PYRAMID],
             file_type=XisfFilePath)
def downscale_master(
    stack: BaseStack, config: ProcessingConfig,
) -> WithSidecar[XisfFilePath]:
    """Block-mean downscale a stack master to the DOWNSCALED_MASTER
    output path. Reads the master via `get_filename_fn(StackId,
    MASTER)`, which resolves to `stack_files`'s output."""
    master_fn = get_filename_fn(
        MediaType.IMAGE, identity_for_stack(stack), MASTER,
    )
    master_path = master_fn(stack)

    output_path = downscale_master.filename_fn(stack)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    factor = DOWNSCALE_FACTORS[DOWNSCALED_MASTER]
    image = read_image(master_path)

    h, w = image.shape[:2]
    new_h, new_w = h // factor, w // factor
    cropped = image[:new_h * factor, :new_w * factor]
    if image.ndim == 2:
        downscaled = cropped.reshape(new_h, factor, new_w, factor).mean(axis=(1, 3))
        # XISF writer expects (h, w, c); promote mono to single-channel.
        downscaled = downscaled[..., np.newaxis]
    else:
        c = image.shape[2]
        downscaled = cropped.reshape(new_h, factor, new_w, factor, c).mean(axis=(1, 3))

    log.info(
        "Downscaled master", stack=str(stack), factor=factor,
        in_shape=image.shape, out_shape=downscaled.shape,
        output=str(output_path),
    )
    metadata = {
        "IMAGETYP": f"Master {stack.image_type}",
        "STBLHASH": stack.stable_hash,
    }
    write_xisf_image(
        output_path, downscaled.astype(np.float32), metadata=metadata,
    )
    return WithSidecar(XisfFilePath(output_path))
