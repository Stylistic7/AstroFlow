"""Final per-channel gradient correction for the assembled mosaic.

The per-panel pipeline already gradient-corrects each panel (MGC, which
uses the MARS all-sky reference + each panel's WCS) and PhotometricMosaic
matches the seams — but PMM with target-gradient correction off only
re-levels each panel by a single average offset beyond the overlap, so a
residual per-panel gradient (measured ~1-2% of sky on broadband, where
light pollution dominates; <0.1% on narrowband) survives the merge and
*accumulates* across panels into a large-scale drift over the finished
stitch.

This op is the standard fix: one gradient-correction pass on the whole
assembled channel, while still linear, before color-combine/stretch. It
is the terminal node of the per-(mosaic, filter) PixInsight pipeline (it
subsumed the old pure-persist `store_mosaic_color_channel_xisf`), so it
carries `final=True` and the displayed `"Mosaic"` tap.

Why GradientCorrection (data-driven) and NOT MGC here, unlike
`post_process_stack`: the assembled mosaic produced by `merge_images` has
NO plate solution (only `IMAGETYP=light` frames are solve-field'd by
`compute_image_metadata`; the merged stitch isn't tagged as one), and
MGC/SPFC both require a WCS (MARS lookup + catalog star matching by sky
coordinates). The MARS correction has already been applied per panel; the
only thing left for this terminal pass is to remove the *accumulated*
residual, for which data-driven GradientCorrection (the modern DBE
replacement — no external reference, no WCS) is exactly the right tool and
the conventional choice for a final whole-mosaic flatten.

Edge erosion before correction: GradientCorrection is documented to be
"sensitive to sudden changes in brightness ... bright or dark artifacts
may occur along the edges. Pure black (zero) pixels at the edges are not a
problem, but very dim, non-zero ... pixels must be cropped out before
running the process." The assembled mosaic has a ragged footprint with a
thin partial-coverage ring of dim non-zero pixels; left in place, GC turns
it into a dark edge band. So we erode the footprint inward to a clean
pure-black boundary first (same `_erode_edges` shave `place_image` applies
per panel), which GC then tolerates.
"""
import os
from pathlib import Path

from pix import Image
from pix.tasks import GradientCorrection
from decorators import node
from decisions.settings import ProcessingConfig
from metadata.classes import ImageMetadata
from mosaics.types import Mosaic
from file_io.types import XisfFilePath, WithSidecar
import structlog

log = structlog.get_logger()


def _eroded_copy(src: Path, dst: Path, px: int) -> None:
    """Write `src` (an XISF mosaic) to `dst` with its ragged footprint
    eroded inward by `px` pixels to a clean pure-black boundary, preserving
    image + file metadata. Mirrors `apply_background_pedestal_to_xisf`'s
    read-modify-write; reuses `place_image`'s `_erode_edges` shave."""
    from xisf import XISF  # type: ignore
    from mosaics.placement.astroflow import _erode_edges

    xisf = XISF(str(src))
    image = xisf.read_image(0)
    file_meta = xisf.get_file_metadata()
    ims_meta = xisf.get_images_metadata()
    # XISF read returns (H, W, C); `_erode_edges` works on the 2-D plane.
    # These are mono per-filter channels (C=1) — erode the plane, restore
    # the channel axis so XISF.write gets the shape it expects.
    if image.ndim == 3:
        image = _erode_edges(image[..., 0], px)[..., None]
    else:
        image = _erode_edges(image, px)
    del xisf
    XISF.write(
        str(dst), image, creator_app="AstroFlow",
        image_metadata=ims_meta[0], xisf_metadata=file_meta,
    )


def _post_process_channel_narrow(args, kwargs):
    """Cache-key hook: this op's `config` dependencies are `mock_pixinsight`
    and `edge_erosion_px`; project `config` to that pair (in its arg slot)
    so the key stays independent of unrelated config fields."""
    mosaic, filter, config, merged = args
    projected = (config.development_settings.mock_pixinsight, config.edge_erosion_px)
    return (mosaic, filter, projected, merged), kwargs


@node(
    display="Mosaic",
    final=True,
    cache_key_narrow=_post_process_channel_narrow,
    resources={"PixInsight": 1},
    auto_memory=True,
)
def post_process_mosaic_color_channel(
    mosaic: Mosaic, filter: str, config: ProcessingConfig,
    merged: ImageMetadata,
) -> WithSidecar[XisfFilePath]:
    """Flatten the assembled per-filter mosaic channel and persist it.

    `merged` is the output of `merge_images`. We erode the ragged footprint
    to a clean pure-black edge, run one data-driven GradientCorrection pass
    to remove the residual large-scale gradient that accumulates across
    panels (see module docstring), then the additive pedestal pins the
    result back to the common sky floor.
    """
    mock_pixinsight = config.development_settings.mock_pixinsight
    output_filename = post_process_mosaic_color_channel.filename_fn(
        mosaic, filter, config, merged,
    )
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    source = merged.filename
    eroded_tmp: Path | None = None
    if not mock_pixinsight:
        # Clean the ragged edge to pure black before GC (see module
        # docstring). Scratch file alongside the output; removed below.
        eroded_tmp = Path(output_filename).with_suffix(".eroded.xisf")
        _eroded_copy(Path(merged.filename), eroded_tmp, config.edge_erosion_px)
        source = eroded_tmp

    image = Image.open(source, filter=filter)
    GradientCorrection(image).save_to(output_filename, mock=mock_pixinsight)

    if eroded_tmp is not None:
        eroded_tmp.unlink(missing_ok=True)

    # Re-equalize the flattened channel's sky to the common pedestal so the
    # final master sits at a known, positive sky floor (same additive step
    # every per-panel master gets in `post_process_stack`). GradientCorrection
    # can shift the absolute level; this pins it back.
    if not mock_pixinsight:
        from stacks.post_processing.background_pedestal import (
            apply_background_pedestal_to_xisf,
        )
        apply_background_pedestal_to_xisf(Path(output_filename))

    return WithSidecar(XisfFilePath(output_filename))
