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


def _post_stack_calibrate_narrow(args, kwargs):
    """Cache-key hook: this op depends on `config` only through
    `mock_pixinsight`, `background_pedestal`, and `background_extraction`;
    project the key to that triple and stay independent of unrelated
    config churn."""
    stack, config, master = args
    narrowed = (config.development_settings.mock_pixinsight,
                config.background_pedestal,
                config.background_extraction)
    return (stack, narrowed, master), kwargs


# The photometric gauge (see units.py): calibrated masters are physical
# SPFC band flux scaled by SPFC_REFERENCE_SCALE — an engineering unit,
# DECLARED on every output via PHOTREF (HST keyword pattern; per-band
# PHOTNMGY conversion to genuine SDSS nanomaggies is the follow-up).
from units import SPFC_REFERENCE_SCALE, SpfcScale  # noqa: E402

_SPFC_REFERENCE_SCALE: SpfcScale = SpfcScale(SPFC_REFERENCE_SCALE)

# Reads the SPFC solution off the live view (PI 1.9.4 publishes it as the
# PCL:SPFC:* properties, NOT the old PSFFLX00 keyword), prints the
# certificate for the sidecar, and applies the gain in-session.
# rescale/truncate stay OFF: the correction must be purely linear, and
# star cores above 1.0 are legitimate in float.
_SPFC_GAIN_JS = """
let view = View.viewById("{view_id}");
function first(v) {{
    return (typeof v == "number") ? v : (v && (v.at ? v.at(0) : v[0]));
}}
let flx = first(view.propertyValue("PCL:SPFC:ScaleFactors"));
let n = first(view.propertyValue("PCL:SPFC:Counts"));
let sig = first(view.propertyValue("PCL:SPFC:Sigmas"));
if (flx == null || !isFinite(flx) || flx <= 0)
    throw new Error("post_stack_calibrate: no SPFC scale after SFC");
let scale = {reference} / flx;
console.writeln(format(
    "ASTROFLOW: spfc_cert scale %.8e n %d sigma %.8e gain %.6f",
    flx, n | 0, sig || 0, scale));
let P = new PixelMath;
P.expression = format("$T*%.10f", scale);
P.useSingleExpression = true;
P.createNewImage = false;
P.rescale = false;
P.truncate = false;
P.executeOn(view);
"""


@node(
    display="Post-Processed",
    cache_key_narrow=_post_stack_calibrate_narrow,
    resources={"PixInsight": 1},
    auto_memory=True,
)
def post_stack_calibrate(
    stack: Stack, config: ProcessingConfig, master: ImageMetadata,
) -> WithSidecar[XisfFilePath]:
    """Post-stacking calibration: the WBPP master made merge-ready.

    ONE persisted artifact per stack (Stylistic7, 2026-09-01 — no cached copy
    whose only delta is a scalar divide): Gaia gain division (SPFC — the
    pipeline's only externally ground-truthed step; gains BEFORE
    backgrounds), then GraXpert background extraction (gated by
    `config.background_extraction`), then the sky pinned to the shared
    pedestal. Output = gain-true, gradient-corrected, pedestal-pinned —
    and worth eyeballing under the Post-Processed tap.

    The SPFC certificate (scale, N stars, sigma, reference gauge) is
    recorded in the output's sidecar. A master SPFC cannot calibrate
    (too few matchable stars — clouds, starvation, bad pointing) is
    REFUSED loudly: something is wrong with it, and known-bad data must
    not merge; quarantine or reshoot.

    Pixels are written back into a copy of the gain-true XISF so FITS
    keywords and the astrometric solution survive (without that,
    compute_image_metadata would re-solve from scratch)."""
    if master.plate_solution is None:
        # compute_image_metadata already raises when solve-field fails on
        # a light frame, so a master with no solution here is an upstream
        # bug. Fail loud.
        raise ValueError(
            f"post_stack_calibrate: master {master.filename} has no "
            f"plate_solution; cannot inject WCS for SPFC"
        )

    output_filename = post_stack_calibrate.filename_fn(stack, config, master)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    mock = config.development_settings.mock_pixinsight
    if mock:
        _write_pixels_preserving_metadata(
            Path(master.filename), Path(output_filename), None,
        )
        return WithSidecar(XisfFilePath(output_filename))

    import re

    from file_io.conversion import convert_xisf_to_fits
    from file_io.generic import read_image
    from file_io.sidecar import set_section
    from pix.image import Image
    from pix.recipe import PixInsightError, Step
    from pix.tasks import (
        SpectrophotometricFluxCalibration,
        WriteAstrometricSolution,
    )

    with tempfile.TemporaryDirectory(prefix="post_stack_calibrate_") as td:
        gain_xisf = Path(td) / "gain_true.xisf"
        img = Image.open(Path(master.filename), filter=stack.filter)
        img = WriteAstrometricSolution(img, master.plate_solution.header)
        img = SpectrophotometricFluxCalibration(img)
        img.recipe.add_step(Step(
            name="spfc gain",
            javascript=_SPFC_GAIN_JS.format(
                view_id=img.view_id, reference=_SPFC_REFERENCE_SCALE),
        ))
        try:
            img.save_to(gain_xisf)
        except PixInsightError as e:
            if "Insufficient data" not in str(e):
                raise
            raise ValueError(
                f"post_stack_calibrate: SPFC found too few calibratable "
                f"stars on {stack.stable_hash} filter {stack.filter} "
                f"({master.filename}). REFUSING to pass uncalibrated data "
                f"downstream — quarantine or reshoot this panel."
            ) from e

        m = re.search(
            r"spfc_cert scale ([\d.eE+-]+) n (\d+) sigma ([\d.eE+-]+)",
            img.recipe.log or "")
        cert = {
            "scale": float(m.group(1)), "n_stars": int(m.group(2)),
            "sigma": float(m.group(3)),
            "reference": _SPFC_REFERENCE_SCALE,
        } if m else None

        corrected = None
        if config.background_extraction:
            from graxpert_cli import run_background_extraction
            as_fits = Path(td) / "in.fits"
            gx_out = Path(td) / "corrected.fits"
            convert_xisf_to_fits(gain_xisf, as_fits)
            run_background_extraction(
                as_fits, gx_out,
                smoothing=GRAXPERT_SMOOTHING,
                correction=GRAXPERT_CORRECTION,
                ai_version=GRAXPERT_AI_VERSION,
            )
            corrected = np.asarray(read_image(gx_out)).astype(np.float32)
            # GraXpert may emit an (H, W, 1) cube for a mono master —
            # flatten so the buffer matches the XISF it goes back into.
            if corrected.ndim == 3 and corrected.shape[2] == 1:
                corrected = corrected[:, :, 0]

        _write_pixels_preserving_metadata(
            gain_xisf, Path(output_filename), corrected,
        )

    # Equalize the sky to the shared pedestal — adjacent mosaic panels
    # must land on the same baseline or the seam shows.
    from stacks.post_processing.background_pedestal import (
        apply_background_pedestal_to_xisf,
    )
    apply_background_pedestal_to_xisf(
        Path(output_filename), target=config.background_pedestal,
    )

    # Declare the gauge on the artifact itself: pixels are physical SPFC
    # band flux scaled by PHOTREF. Any consumer can convert onward
    # without asking us.
    from file_io.xisf_file_io import update_xisf_header
    update_xisf_header(Path(output_filename), {
        "PHOTREF": float(_SPFC_REFERENCE_SCALE),
    })

    if cert:
        set_section(Path(output_filename), "spfc_calibration", cert)
        log.info("post_stack_calibrate complete", stack=stack.stable_hash,
                 filter=stack.filter, gain=cert["reference"] / cert["scale"],
                 n_stars=cert["n_stars"], output=str(output_filename))
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
