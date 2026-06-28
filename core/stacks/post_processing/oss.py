"""OSS stack post-processing: GraXpert background/gradient correction + solve.

The OSS counterpart to PixInsight's SpectrophotometricFluxCalibration +
MultiscaleGradientCorrection. Gradient correction is done by **GraXpert** (an
open-source AI background extractor), driven as an external CLI — see
`graxpert_cli`. Flux calibration is deferred. PixInsight is never invoked, so
this op works directly on the OSS FITS master (PI's SPFC/MGC need metadata the
OSS master doesn't carry).

GraXpert install (in any environment on PATH, or pointed at by
ASTROFLOW_GRAXPERT_BIN): `pip install "graxpert[cpuonly]"` on Apple Silicon
(the bare `graxpert` is missing the onnxruntime AI library). See `graxpert_cli`.
"""
import os
import tempfile
from pathlib import Path

import numpy as np
import structlog

import astropy.io.fits as pyfits

import plate_solving
from decorators import node
from file_io.generic import read_image
from file_io.types import FitsFilePath, WithSidecar
from stacks.composite_stack import Stack

log = structlog.get_logger()


@node(
    display="Post-Processed",
    auto_memory=True,
)
def post_process_stack_oss(
    stack: Stack, mock_pixinsight: bool, background_pedestal: float, master,
) -> WithSidecar[FitsFilePath]:
    """GraXpert background-correct + plate-solve a stacked OSS master.

    `master` arrives as the upstream stacker's ImageMetadata; its content_hash
    flows into this op's call_hash via `filename_fn(*bound.args)`, so distinct
    upstream masters produce distinct output filenames automatically.

    `background_pedestal` (from `config.background_pedestal`) is the common sky
    level the master is shifted to; it's an explicit arg so changing it
    re-derives this master. `mock_pixinsight=True` (fast tests) skips the heavy
    external GraXpert run and passes the stacked data through with
    post-processed headers — keeping the test suite free of the GraXpert
    install. Production runs GraXpert.
    """
    output_filename = post_process_stack_oss.filename_fn(
        stack, mock_pixinsight, background_pedestal, master,
    )
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    log.info("OSS post-processing stack", stack=str(stack),
             master=str(master.filename), mock=mock_pixinsight)

    if mock_pixinsight:
        data = read_image(master.filename).astype(np.float32)
    else:
        # GraXpert AI background/gradient correction — the OSS counterpart to
        # MultiscaleGradientCorrection (data-driven, no MARS reference). Run as
        # an external CLI to a temp file; read its output back so we own the
        # final headers (GraXpert writes its own).
        from graxpert_cli import run_background_extraction
        with tempfile.TemporaryDirectory(prefix="graxpert_") as td:
            gx_out = Path(td) / "corrected.fits"
            run_background_extraction(master.filename, gx_out)
            data = read_image(gx_out).astype(np.float32)

    # Mono masters: GraXpert may emit an (H, W, 1) cube — flatten to 2-D so
    # downstream metadata/solve/merge see the same shape as the input.
    if data.ndim == 3 and data.shape[2] == 1:
        data = data[:, :, 0]

    # Bring the raw-ADU OSS stack onto the same representation a PI master uses:
    # normalize to the [0,1] convention (16-bit full scale), then equalize the
    # sky to the common pedestal — so OSS and PI masters share a scale and a sky
    # level, and the mosaic blend is seam-free either way. `mask_zeros=False`:
    # this is a solid master (no panel boundary yet), and GraXpert leaves
    # sub-zero sky pixels that must be shifted up, not read as an edge.
    from stacks.post_processing.background_pedestal import apply_background_pedestal
    data = (data / 65535.0).astype(np.float32)
    data = apply_background_pedestal(data, target=background_pedestal, mask_zeros=False)

    header = pyfits.Header()
    header["IMAGETYP"] = "Post-processed Light"
    header["STBLHASH"] = stack.stable_hash
    header["FILTER"] = stack.filter
    header["PIPELINE"] = "AstroFlow-OSS"
    header["RA"] = stack.center[0]
    header["DEC"] = stack.center[1]
    pyfits.PrimaryHDU(data=data, header=header).writeto(str(output_filename), overwrite=True)

    plate_solving.solve_field(output_filename)
    log.info("OSS post-processing complete", output=str(output_filename))
    return WithSidecar(FitsFilePath(output_filename))
