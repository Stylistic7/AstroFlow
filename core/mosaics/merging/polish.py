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
import pyvips
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


def _lazy_polish(img: pyvips.Image, model_small: np.ndarray,
                 shift: float) -> pyvips.Image:
    """LAZY pyvips pipeline: upsample the smooth model to `img`'s grid,
    subtract inside the coverage mask, apply the pedestal `shift`, keep
    covered pixels strictly positive and no-data exactly zero. Pure —
    the testable half of the op. Nothing canvas-sized ever materializes:
    tiffsave pulls this expression tile-by-tile (Stylistic7, 2026-09-02:
    never read the entire mosaic into memory)."""
    h_s, w_s = model_small.shape
    buf = np.ascontiguousarray(model_small, dtype=np.float32)
    # copy_memory() decouples the (tiny, 2k-max) model from the numpy
    # buffer so the lazy pipeline can outlive this frame safely.
    model = pyvips.Image.new_from_memory(
        buf.data, w_s, h_s, 1, "float").copy_memory()
    model = model.resize(img.width / w_s, vscale=img.height / h_s,
                         kernel="linear")
    # resize rounds output dims; normalize to img's grid exactly (the
    # sliver is <=1 px of a smooth field — edge-copy is exact enough).
    model = model.crop(0, 0, min(model.width, img.width),
                       min(model.height, img.height)) \
                 .embed(0, 0, img.width, img.height, extend="copy")
    tiny = float(np.finfo(np.float32).tiny)
    corrected = img - model + float(shift)
    corrected = (corrected < tiny).ifthenelse(tiny, corrected)
    return (img > 0).ifthenelse(corrected, 0.0).cast("float")


# auto_memory dropped with the streaming rewrite (Stylistic7, 2026-09-02):
# the op's peak RSS is now ~the 2k model, so a profiled reservation
# would only re-teach the scheduler yesterday's ~3x-canvas appetite.
@node(display="Mosaic", final=True, plate_solve=False)  # mosaic projection — non-square pixels
def polish_background(
    mosaic: Mosaic, filter: str, merged: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """The channel deliverable: `merged` minus its global background model,
    sky restored to the house pedestal. Presented as the merge-polish
    scope's terminal, so this op's output IS the "Mosaic" tap and its
    source hash is what the colour tail's panel manifests fold in."""
    from file_io.tiff_file_io import write_tiff_image_streaming
    from stacks.post_processing.background_pedestal import DEFAULT_PEDESTAL

    output_filename = polish_background.filename_fn(mosaic, filter, merged)
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)

    src = Path(merged.filename).resolve(strict=True)
    img = pyvips.Image.new_from_file(str(src), unlimited=True)
    if img.bands > 1:
        img = img[0]

    # Small copy via a COVERAGE-NORMALIZED streaming box shrink. A plain
    # resize both averages no-data zeros into every boundary pixel
    # (darkening bias) and — with vips' default lanczos kernel — RINGS on
    # the coverage step edge, which GraXpert then bakes into the model as
    # a periodic bright/dark comb along every edge ("warbling", Stylistic7,
    # 2026-09-02). Box-shrinking image and mask separately and dividing
    # makes each small pixel the mean of ONLY its covered full-res pixels
    # — equivalent to a locally-exact fill before the downscale — with a
    # ringless kernel. Small pixels under 25% coverage go to zero and are
    # median-filled inside _background_model_small as before.
    k = max(img.width, img.height) / _MODEL_INPUT_PX
    cov = (img > 0).cast("float") / 255.0  # vips relational is 0/255
    small_sum = img.shrink(k, k).numpy().astype(np.float32)
    small_cov = cov.shrink(k, k).numpy().astype(np.float32)
    if small_sum.ndim == 3:
        small_sum = small_sum[..., 0]
    if small_cov.ndim == 3:
        small_cov = small_cov[..., 0]
    small = np.where(small_cov >= 0.25,
                     small_sum / np.maximum(small_cov, 1e-6),
                     np.float32(0))
    model_small = _background_model_small(small)

    # Pedestal shift measured at model scale: the corrected field is
    # smooth there, and the median of the area-reduced image tracks the
    # full-res masked median to ~1e-5 — the pedestal is a shared
    # baseline convention, not photometry, and every channel uses the
    # identical estimator.
    mask_small = small > 0
    sky = float(np.median((small - model_small)[mask_small])) \
        if mask_small.any() else 0.0
    shift = float(DEFAULT_PEDESTAL) - sky

    result = _lazy_polish(img, model_small, shift)
    write_tiff_image_streaming(Path(output_filename), result,
                               {"FILTER": filter,
                                "STBLHASH": str(merged.stable_hash),
                                "PIPELINE": "AstroFlow-OSS",
                                "POLISHVER": "gx-mono-3-covnorm"})
    _set_panel_count(output_filename, _panel_count(merged))

    log.info("background polish complete", output=str(output_filename),
             filter=filter)
    return WithSidecar(output_filename)
