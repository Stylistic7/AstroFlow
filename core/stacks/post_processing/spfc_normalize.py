"""SPFC gain normalization — the multiplicative gauge, corrected at the source.

Measured on the Cygnus 24-panel mosaic (2026-08-20, triple-reproduced):
straight out of WBPP, sibling H masters differ in true flux scale by 14%
(MAD/median) with a 1.79x worst pair — night-to-night extinction/airmass
drift plus WBPP's per-stack reference gauge. Nothing downstream corrects a
GAIN: GraXpert subtracts backgrounds (additive), mBgModel solves levels
(additive), and the OSS blend's flux solve is diluted to ~unity by the
GraXpert-equalized sky in its overlap means. The multiplicative error rides
the SIGNAL all the way into the merge and quilts the emission.

This op divides it out where it is measurable: SpectrophotometricFluxCalibration
against Gaia DR3/SP star photometry, on the master directly out of
`stack_files`, then PixelMath `$T x (PSFFLX00 / reference)` — after which
every panel's signal sits on ONE absolute scale and the remaining
cross-panel disagreement is purely additive, exactly what the downstream
correctors assume.

The reference is a FIXED constant, not a per-mosaic statistic: SPFC is
absolute (Gaia), so no joint solve is needed, and a joint reference would
re-key every sibling whenever one panel changed. Only ratios matter; 2600
sits at the observed H median so numeric ranges stay familiar downstream
(pedestal ~0.005, GraXpert contrast regime unchanged).

Sky note: the rescale multiplies each panel's (still-present) sky by its
own factor — deliberately fine. The physical sky truly varied 2.2x across
the Cygnus panels (uncorrelated with the gains, r = -0.07); it must be
SUBTRACTED per panel, and GraXpert does that next, on data whose signal is
now gain-true.
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

# Fixed gain reference (image units per PSFFLX00 unit). Arbitrary gauge —
# only ratios matter — chosen at the observed Cygnus H median so normalized
# masters land in the numeric range the rest of the pipeline already
# expects. Changing it re-keys every normalized master; don't.
_REFERENCE_FLUX = 2600.0

# Reads the solved PSFFLX00 off the view and applies the gain in-session.
# rescale/truncate stay OFF: the correction must be purely linear, and
# star cores above 1.0 are legitimate in float.
_APPLY_GAIN_JS = """
let view = View.viewById("{view_id}");
let flx = null;
let kws = view.window.keywords;
for (let i = 0; i < kws.length; ++i)
    if (kws[i].name.trim() == "PSFFLX00")
        flx = parseFloat(kws[i].value);
if (flx == null || !isFinite(flx) || flx <= 0)
    throw new Error("spfc_normalize: no valid PSFFLX00 after SFC");
let scale = flx / {reference};
console.writeln(format(
    "ASTROFLOW: spfc_normalize gain %.6f (PSFFLX00 %.1f / ref %.1f)",
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
            view_id=img.view_id, reference=_REFERENCE_FLUX),
    ))
    img.save_to(
        output_filename, mock=config.development_settings.mock_pixinsight,
    )
    log.info("spfc_normalize complete", stack=stack.stable_hash,
             filter=stack.filter, output=str(output_filename))
    return WithSidecar(XisfFilePath(output_filename))
