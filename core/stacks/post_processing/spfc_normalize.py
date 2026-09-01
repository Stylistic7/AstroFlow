"""SPFC gain normalization — the multiplicative gauge, corrected at the source.

RESTORED 2026-09-01 (deleted with the Montage arms in 0b0c98b3) after the
mosaic labs validated the whole loop end to end: SPFC recovers injected
per-panel gains to <=0.8% even on shot-noise-limited panels
(mosaics/test_synthetic_mosaic_lab.py), and it is the ONLY step in the
pipeline anchored to external reality (Gaia DR3/SP spectrophotometry) —
every other correction is internal self-consistency. Kept per Stylistic7's
decision: "it's literally the only ground-truthing in the entire pipeline."

Measured on the Cygnus 24-panel mosaic (2026-08-20, triple-reproduced):
straight out of WBPP, sibling H masters differ in true flux scale by 14%
(MAD/median) with a 1.79x worst pair. Nothing downstream corrects a GAIN
at the source: the merge's overlap flux solve estimates it pairwise from
overlap content (scene-dependent, panel-count-scaling failure statistics);
GraXpert and the level solve are additive. This op divides the gain out
where it is measurable — against Gaia — after which panel disagreement is
purely additive, exactly what the downstream correctors assume.

Changes from the original (all lab-verified 2026-09-01):
  * PI 1.9.4 SPFC publishes the scale as the view property
    `PCL:SPFC:ScaleFactors` (data-units per predicted-physical-flux;
    RISES with gain). The old PSFFLX00 keyword path is kept as fallback.
  * The reference is a FIXED constant in ScaleFactors units. Only ratios
    matter (absolute gauge is arbitrary); it pins normalized masters to a
    familiar numeric range. CALIBRATE against one real master before the
    first production deploy (run SPFC on a Cygnus H master, set the
    observed median here) — the placeholder is the lab-typical value.
"""

import os

from pix import Image
from pix.recipe import Step
from pix.tasks import SpectrophotometricFluxCalibration, WriteAstrometricSolution
from decorators import node
from decisions.settings import ProcessingConfig
from metadata.classes import ImageMetadata
from stacks.composite_stack import Stack
from file_io.types import XisfFilePath, WithSidecar
import structlog

log = structlog.get_logger()

# Fixed gain reference in PCL:SPFC:ScaleFactors units (data per physical).
# Arbitrary gauge — only ratios matter. TODO(production deploy): calibrate
# on a real Cygnus master and freeze; changing it re-keys every normalized
# master.
_REFERENCE_SCALE = 0.15

# Reads the solved scale off the view and applies the gain in-session.
# rescale/truncate stay OFF: the correction must be purely linear, and
# star cores above 1.0 are legitimate in float.
_APPLY_GAIN_JS = """
let view = View.viewById("{view_id}");
let flx = null;
let sf = view.propertyValue("PCL:SPFC:ScaleFactors");
if (sf != null)
    flx = (typeof sf == "number") ? sf : (sf.at ? sf.at(0) : sf[0]);
if (flx == null || !isFinite(flx) || flx <= 0) {{
    let kws = view.window.keywords;
    for (let i = 0; i < kws.length; ++i)
        if (kws[i].name.trim() == "PSFFLX00")
            flx = parseFloat(kws[i].value);
}}
if (flx == null || !isFinite(flx) || flx <= 0)
    throw new Error("spfc_normalize: no SPFC scale after SFC");
let scale = {reference} / flx;
console.writeln(format(
    "ASTROFLOW: spfc_normalize gain %.6f (scale %.6e / ref %.6e)",
    scale, flx, {reference}));
let P = new PixelMath;
P.expression = format("$T*%.10f", scale);
P.useSingleExpression = true;
P.createNewImage = false;
P.rescale = false;
P.truncate = false;
P.executeOn(view);
"""


def _spfc_normalize_narrow(args, kwargs):
    """Cache-key hook mirroring the sibling post-process ops: this op
    depends on `config` only through `mock_pixinsight`, so project the key
    to that and stay independent of unrelated config churn."""
    stack, config, master = args
    return (stack, config.development_settings.mock_pixinsight, master), kwargs


@node(
    display="Flux-Normalized",
    cache_key_narrow=_spfc_normalize_narrow,
    resources={"PixInsight": 1},
    auto_memory=True,
)
def spfc_normalize(
    stack: Stack, config: ProcessingConfig, master: ImageMetadata,
) -> WithSidecar[XisfFilePath]:
    """The WBPP master with its Gaia gain divided out (see module doc)."""
    if master.plate_solution is None:
        # Same fail-loud contract as post_process_stack: a master without a
        # WCS reaching this op is an upstream bug, and SFC cannot run
        # without one.
        raise ValueError(
            f"spfc_normalize: master {master.filename} has no "
            f"plate_solution; cannot inject WCS for SFC"
        )

    output_filename = spfc_normalize.filename_fn(stack, config, master)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    img = Image.open(master.filename, filter=stack.filter)
    img = WriteAstrometricSolution(img, master.plate_solution.header)
    img = SpectrophotometricFluxCalibration(img)
    img.recipe.add_step(Step(
        name="spfc_normalize gain",
        javascript=_APPLY_GAIN_JS.format(
            view_id=img.view_id, reference=_REFERENCE_SCALE),
    ))
    img.save_to(
        output_filename, mock=config.development_settings.mock_pixinsight,
    )
    log.info("spfc_normalize complete", stack=stack.stable_hash,
             filter=stack.filter, output=str(output_filename))
    return WithSidecar(XisfFilePath(output_filename))
