"""GraXpert post-processing of a WBPP-stacked master.

The A/B counterpart to `stacks.post_processing.pixinsight.post_process_stack`:
same inputs, same output contract, but gradient correction is done by
**GraXpert** instead of PixInsight's SpectrophotometricFluxCalibration +
MultiscaleGradientCorrection.

WHY, measured on the Heart Nebula's OIII stack (2026-08-14). Edge-vs-interior
background deviation on the raw WBPP master, and flux retained above local sky
in the brightest 1% of pixels:

    variant                    bowl    signal kept
    raw (no correction)       6.11%         100.0%
    SPFC + MGC (pipeline)     0.90%          91.5%
    GraXpert only (this)      0.24%          91.6%
    MGC then GraXpert         0.19%          91.2%

MGC removes 85% of the bowl; GraXpert alone removes 96%, and loses no more
signal doing it. MGC's residual is a symmetric four-sided bowl — a FLAT-FIELD
error, which a MARS *sky* reference cannot model (MARS is 22.7 as/px against
this image's 2.21, from 4 reference frames of total weight 3.52). GraXpert,
being data-driven, fits it directly.

The tradeoff this trades away: MGC is objective — it references real sky — so
on a field with a genuine large-scale sky gradient it will not mistake
nebulosity for background, while a data-driven fit might. The Heart is a poor
test of that (its sky gradient is small), so this pipeline is a deliberate
experiment, not a replacement. See `pipelines/hybrid_graxpert_oss_mosaic.py`.

Settings come from a sweep of GraXpert's full option set on the same image:
smoothing 0.25 was optimal (0.0 → 0.29%, 0.25 → 0.27%, 0.5 → 0.31%,
0.75 → 0.44%, 1.0 → 0.57%); Subtraction and Division were indistinguishable
(0.27% vs 0.28%) at this amplitude, so Subtraction wins for leaving the
pedestal untouched; and AI model 1.0.1 beat 1.0.0 decisively (0.27% vs 0.61%),
so it is PINNED rather than left as "latest".

No PixInsight slot is taken — GraXpert is an external CLI — so these stacks no
longer contend for the 2 PI slots.
"""
import os
import tempfile
from pathlib import Path

import numpy as np
import structlog

from decorators import node
from decisions.settings import ProcessingConfig
from file_io.types import XisfFilePath, WithSidecar
from metadata.classes import ImageMetadata
from stacks.composite_stack import Stack

log = structlog.get_logger()

# Pinned from the 2026-08-14 sweep; see module docstring.
GRAXPERT_SMOOTHING = 0.25
GRAXPERT_CORRECTION = "Subtraction"
GRAXPERT_AI_VERSION = "1.0.1"


def _post_process_graxpert_narrow(args, kwargs):
    """Cache-key hook mirroring `pixinsight._post_process_narrow`: this op
    depends on `config` only through `mock_pixinsight` and
    `background_pedestal`, so project it to that pair and keep the key
    independent of unrelated config churn."""
    stack, config, master = args
    narrowed = (config.development_settings.mock_pixinsight,
                config.background_pedestal)
    return (stack, narrowed, master), kwargs


@node(
    display="Post-Processed",
    cache_key_narrow=_post_process_graxpert_narrow,
    auto_memory=True,
)
def post_process_stack_graxpert(
    stack: Stack, config: ProcessingConfig, master: ImageMetadata,
) -> WithSidecar[XisfFilePath]:
    """GraXpert background/gradient correction of a stacked master.

    Contract-identical to `post_process_stack` — same args, same
    `WithSidecar[XisfFilePath]` result, same `display="Post-Processed"` tap —
    so every downstream op (merge_target_channel, place_image, the colour
    tail) consumes it unchanged.

    The corrected pixels are written back into a COPY OF THE ORIGINAL XISF
    rather than into GraXpert's own FITS output, so the master keeps its FITS
    keywords and XISF metadata (FILTER, STBLHASH, the WCS `post_process_stack`
    would otherwise have injected). Without that, `compute_image_metadata`
    would find no plate solution on the output and re-solve it from scratch.
    """
    if master.plate_solution is None:
        # Same invariant as post_process_stack: compute_image_metadata already
        # raises when solve-field fails on a light frame, so a master with no
        # solution reaching this op is an upstream bug. Fail loud.
        raise ValueError(
            f"post_process_stack_graxpert: master {master.filename} has no "
            f"plate_solution; cannot post-process"
        )

    output_filename = post_process_stack_graxpert.filename_fn(
        stack, config, master,
    )
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    from file_io.conversion import convert_xisf_to_fits
    from file_io.generic import read_image

    mock = config.development_settings.mock_pixinsight
    corrected = None
    if not mock:
        from graxpert_cli import run_background_extraction
        with tempfile.TemporaryDirectory(prefix="graxpert_stack_") as td:
            as_fits = Path(td) / "in.fits"
            gx_out = Path(td) / "corrected.fits"
            convert_xisf_to_fits(Path(master.filename), as_fits)
            run_background_extraction(
                as_fits, gx_out,
                smoothing=GRAXPERT_SMOOTHING,
                correction=GRAXPERT_CORRECTION,
                ai_version=GRAXPERT_AI_VERSION,
            )
            corrected = np.asarray(read_image(gx_out)).astype(np.float32)
        # GraXpert may emit an (H, W, 1) cube for a mono master — flatten so
        # the buffer matches the XISF we are writing it back into.
        if corrected.ndim == 3 and corrected.shape[2] == 1:
            corrected = corrected[:, :, 0]

    _write_pixels_preserving_metadata(
        Path(master.filename), Path(output_filename), corrected,
    )

    # Equalize the sky to a known pedestal, exactly as post_process_stack does
    # — adjacent mosaic panels must land on the same baseline or the seam shows.
    if not mock:
        from stacks.post_processing.background_pedestal import (
            apply_background_pedestal_to_xisf,
        )
        apply_background_pedestal_to_xisf(
            Path(output_filename), target=config.background_pedestal,
        )

    return WithSidecar(XisfFilePath(output_filename))


def _write_pixels_preserving_metadata(
    source: Path, dest: Path, pixels: "np.ndarray | None",
) -> None:
    """Write `source`'s XISF to `dest`, swapping in `pixels` when given.

    Same read-modify-write shape as
    `background_pedestal.apply_background_pedestal_to_xisf`: the file and image
    metadata come from the source, so FITS keywords and the astrometric
    solution survive a correction that round-tripped through FITS.
    `pixels=None` (mock mode) copies the master through untouched.
    """
    from xisf import XISF  # type: ignore

    xisf = XISF(str(source))
    image = xisf.read_image(0)
    file_meta = xisf.get_file_metadata()
    ims_meta = xisf.get_images_metadata()
    del xisf

    if pixels is not None:
        if pixels.shape != image.shape[:pixels.ndim]:
            # A shape change would silently corrupt the WCS, so refuse rather
            # than write a master whose astrometry no longer matches its pixels.
            raise ValueError(
                f"GraXpert changed the image shape: {image.shape} -> "
                f"{pixels.shape}; refusing to write {dest}"
            )
        image = pixels.reshape(image.shape).astype(image.dtype, copy=False)

    XISF.write(
        str(dest),
        image,
        creator_app="AstroFlow",
        image_metadata=ims_meta[0],
        xisf_metadata=file_meta,
    )
