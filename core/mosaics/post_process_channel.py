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
NO plate solution (nothing solves it — a reprojected stitch has
non-square pixels and can't be solve-field'd), and MGC/SPFC
both require a WCS (MARS lookup + catalog star matching by sky
coordinates). The MARS correction has already been applied per panel; the
only thing left for this terminal pass is to remove the *accumulated*
residual, for which data-driven GradientCorrection (the modern DBE
replacement — no external reference, no WCS) is exactly the right tool and
the conventional choice for a final whole-mosaic flatten.

No edge erosion here: every panel was already edge-eroded to a pure-black
boundary by `place_image` before the merge, and PhotometricMosaic feathers
only the interior panel *overlaps*, not the outer frame — so the assembled
stitch already presents GradientCorrection the clean zero edge it wants.
(An earlier version re-eroded the whole mosaic preventively "per PI docs";
that just re-shaved an already-clean boundary, discarding valid data.)
"""
from pathlib import Path

from pix import Image
from pix.tasks import GradientCorrection
from decorators import node
from decisions.settings import ProcessingConfig
from metadata.classes import ImageMetadata
from mosaics.types import Mosaic
from file_io.types import Produced, XisfFilePath
import structlog

log = structlog.get_logger()


def _post_process_channel_narrow(args, kwargs):
    """Cache-key hook: this op's only `config` dependency is `mock_pixinsight`;
    project `config` to it (in its arg slot) so the key stays independent of
    unrelated config fields."""
    mosaic, filter, config, merged = args
    projected = config.development_settings.mock_pixinsight
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
    merged: ImageMetadata, *, out: XisfFilePath,
) -> Produced:
    """Flatten the assembled per-filter mosaic channel and persist it.

    `merged` is the output of `merge_images`. One data-driven
    GradientCorrection pass removes the residual large-scale gradient that
    accumulates across panels (see module docstring); the additive pedestal
    then pins the result back to the common sky floor. No edge erosion is
    needed — the panels were already eroded by `place_image`, so the stitch
    reaches GC with a clean pure-black boundary.
    """
    mock_pixinsight = config.development_settings.mock_pixinsight

    image = Image.open(merged.filename, filter=filter)
    GradientCorrection(image).save_to(out, mock=mock_pixinsight)

    # Re-equalize the flattened channel's sky to the common pedestal so the
    # final master sits at a known, positive sky floor (same additive step
    # every per-panel master gets in `post_process_stack`). GradientCorrection
    # can shift the absolute level; this pins it back.
    if not mock_pixinsight:
        from stacks.post_processing.background_pedestal import (
            apply_background_pedestal_to_xisf,
        )
        apply_background_pedestal_to_xisf(Path(out))

    return Produced(out)
