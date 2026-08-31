"""Whole-canvas background polish for merged mosaic channels.

After the overlap-constants level solve, the residual mismatch between
panels is a SMOOTH per-panel model error (GraXpert's per-stack extraction
leaves a slightly different low-order surface in each panel) plus any
genuine mosaic-scale gradient. Both are low-frequency, so one global
background model over the merged canvas removes them — per channel,
independently.

WHY DOWNSCALED: GraXpert's AI background model is computed at 256x256
internally (verified in graxpert/background_extraction.py — the input is
cv2.resized to 256 before the net, and the model resized back up), so
feeding it the gigapixel canvas buys nothing. We hand it a ~2k downscale,
recover the MODEL as (input - output) at small size, upsample the model
(bilinear on a smooth field is lossless), and subtract at full
resolution.

WHY PER-CHANNEL (mono): the net is 3-channel inside — mono input is
replicated across the three net channels. Feeding it a true SHO triple
was measured WORSE (2026-08-31): its cross-channel priors come from
broadband RGB and painted chromatic blotches along the coverage
boundary. Mono-replicated is the validated mode.

ZERO HANDLING: the canvas is mostly no-data (ragged coverage). GraXpert
reads zeros as background (the 14x over-subtraction lesson,
2026-08-12), so no-data pixels are median-filled before modelling and
re-zeroed after. The correction applies only inside the coverage mask.
"""
from pathlib import Path

import numpy as np
import structlog

from decorators.decorators import node
from file_io.tiff_file_io import write_tiff_image
from file_io.types import TiffFilePath, WithSidecar
from metadata.classes import ImageMetadata
from mosaics.merging.pixinsight import _panel_count, _set_panel_count
from mosaics.types import MosaicLike as Mosaic

log = structlog.get_logger()

# The model is inherently 256px (see module docstring); 2048 leaves a wide
# margin for the resize chain while keeping the GraXpert round-trip seconds.
_MODEL_INPUT_PX = 2048


def _background_model_small(small: np.ndarray) -> np.ndarray:
    """GraXpert's smooth background model of `small` (2D float32, zeros =
    no data), at `small`'s own size. Median-fills no-data before the CLI
    call; the returned model is meaningful only inside the coverage mask."""
    import tempfile
    from astropy.io import fits as pyfits
    from graxpert_cli import run_background_extraction
    from stacks.post_processing.graxpert_stack import (
        GRAXPERT_AI_VERSION, GRAXPERT_CORRECTION, GRAXPERT_SMOOTHING,
    )

    mask = small > 0
    filled = np.where(mask, small, np.float32(np.median(small[mask])))
    with tempfile.TemporaryDirectory(prefix="polish_bg_") as td:
        fin = Path(td) / "in.fits"
        fout = Path(td) / "out.fits"
        pyfits.PrimaryHDU(filled.astype(np.float32)).writeto(fin)
        run_background_extraction(
            fin, fout,
            smoothing=GRAXPERT_SMOOTHING,
            correction=GRAXPERT_CORRECTION,
            ai_version=GRAXPERT_AI_VERSION,
        )
        out = pyfits.getdata(fout).astype(np.float32)
    if out.ndim == 3:
        out = out[0] if out.shape[0] in (1, 3) else out[..., 0]
    return filled - out


def _apply_model(full: np.ndarray, model_small: np.ndarray,
                 pedestal: float) -> np.ndarray:
    """Upsample `model_small` to `full`'s size, subtract inside the
    coverage mask, and shift so the masked median sky lands on `pedestal`.
    No-data zeros stay exactly zero; covered pixels stay > 0. Pure — the
    testable half of the op."""
    import cv2

    mask = full > 0
    model = cv2.resize(model_small, dsize=(full.shape[1], full.shape[0]),
                       interpolation=cv2.INTER_LINEAR)
    out = np.where(mask, full - model, np.float32(0))
    sky = float(np.median(out[mask])) if mask.any() else 0.0
    shift = np.float32(pedestal - sky)
    if shift != 0:
        np.add(out, shift, out=out, where=mask)
    np.maximum(out, np.finfo(np.float32).tiny, out=out, where=mask)
    return out


@node(display="Mosaic", final=True, auto_memory=True, plate_solve=False)  # mosaic projection — non-square pixels
def polish_background(
    mosaic: Mosaic, filter: str, merged: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """The channel deliverable: `merged` minus its global background model,
    sky restored to the house pedestal. Presented as the merge-polish
    scope's terminal, so this op's output IS the "Mosaic" tap and its
    source hash is what the colour tail's panel manifests fold in."""
    from file_io.generic import read_image
    from stacks.post_processing.background_pedestal import DEFAULT_PEDESTAL

    output_filename = polish_background.filename_fn(mosaic, filter, merged)
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)

    src = Path(merged.filename).resolve(strict=True)
    full = np.asarray(read_image(src), dtype=np.float32)
    if full.ndim == 3:
        full = full[..., 0]

    scale = _MODEL_INPUT_PX / max(full.shape)
    import cv2
    small = cv2.resize(full, dsize=None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    model_small = _background_model_small(small)
    result = _apply_model(full, model_small, float(DEFAULT_PEDESTAL))

    write_tiff_image(Path(output_filename), result,
                     {"FILTER": filter,
                      "STBLHASH": str(merged.stable_hash),
                      "PIPELINE": "AstroFlow-OSS",
                      "POLISHVER": "gx-mono-1"})
    _set_panel_count(output_filename, _panel_count(merged))

    log.info("background polish complete", output=str(output_filename),
             filter=filter)
    return WithSidecar(output_filename)
