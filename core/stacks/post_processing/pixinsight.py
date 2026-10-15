from pathlib import Path

from pix import Image
from pix.tasks import (
    WriteAstrometricSolution,
    SpectrophotometricFluxCalibration,
    MultiscaleGradientCorrection,
    GradientCorrection,
    AutomaticBackgroundExtractor,
)
from pix.recipe import PixInsightNoReferenceDataError
from pix.tasks.filters import normalize_filter
from decorators import node
from decisions.settings import ProcessingConfig
from metadata.classes import ImageMetadata
from stacks.composite_stack import Stack
from file_io.types import Produced, XisfFilePath
import structlog

log = structlog.get_logger()


def _post_process_narrow(args, kwargs):
    """Cache-key hook: this op depends on `config` only through the
    `mock_pixinsight` flag and the `background_pedestal` level, so project
    `config` to that pair (in its arg slot). Keeps the key independent of
    unrelated config fields while still re-deriving the master when the
    pedestal changes."""
    stack, config, master = args
    narrowed = (config.development_settings.mock_pixinsight, config.background_pedestal)
    return (stack, narrowed, master), kwargs


@node(
    display="Post-Processed",
    cache_key_narrow=_post_process_narrow,
    resources={"PixInsight": 1},
    auto_memory=True,
)
def post_process_stack(
    stack: Stack, config: ProcessingConfig, master: ImageMetadata, *, out: XisfFilePath,
) -> Produced:
    """PixInsight post-processing of a stacked master.

    `master` arrives as an `ImageMetadata` (the @makes_image trace
    hook chains `compute_image_metadata` after every produced
    artifact, so by the time the master reaches this op at runtime
    it carries width_pixels, height_pixels, plate_solution, etc.).
    The body reads everything it needs from `master` and never
    needs to know HOW the master was produced — full-resolution
    stacking, downscaled stacking, or any future variant all look
    identical from here.

    Pipeline: inject solve-field's WCS into PI's native astrometric
    solution format, then run SPCC + MultiscaleGradientCorrection
    (or GradientCorrection for S filter, which has no MARS
    reference data).

    We deliberately do NOT run PI's `ImageSolver` — its pair-RANSAC
    star matching fails deterministically on dense Milky Way fields
    after the 8× downscale, while astrometry.net's `solve-field`
    (which `compute_image_metadata` already ran on this master and
    cached as `master.plate_solution`) uses quad-hashing and handles
    those fields trivially. Reusing that solution is both faster
    (~10s vs 30-60s for ImageSolver+retries) and strictly more
    reliable: any master that reached this op is, by construction,
    one solve-field successfully plate-solved.
    """
    if master.plate_solution is None:
        # `compute_image_metadata` already raises PlateSolveFailedException
        # when solve-field can't solve a light frame, so a master with no
        # plate solution reaching this op would be a bug upstream. Fail
        # loud rather than producing a master without a WCS.
        raise ValueError(
            f"post_process_stack: master {master.filename} has no "
            f"plate_solution; cannot inject WCS"
        )

    mock_pixinsight = config.development_settings.mock_pixinsight

    def _calibrated_master() -> Image:
        """Fresh master view with WCS injected + flux-calibrated. Re-opened
        per attempt so a failed recipe's accumulated steps don't leak into
        the fallback."""
        img = Image.open(master.filename, filter=stack.filter)
        img = WriteAstrometricSolution(img, master.plate_solution.header)
        return SpectrophotometricFluxCalibration(img)

    if normalize_filter(stack.filter) not in ("H", "O", "R", "G", "B", "L"):
        # Filters SPFC/MGC can't serve take the classic data-driven
        # GradientCorrection directly:
        #   * SII — no MARS reference data at all;
        #   * 'none'/unknown — filterless captures (e.g. six M31 frames
        #     from 2026-01-16 whose headers carry no FILTER card):
        #     SpectrophotometricFluxCalibration REJECTS FILTER='none'
        #     outright, and that ValueError silently poisoned census
        #     soundness on every pass until the offender diagnostic
        #     named it (2026-08-10).
        image = Image.open(master.filename, filter=stack.filter)
        image = WriteAstrometricSolution(image, master.plate_solution.header)
        GradientCorrection(image).save_to(out, mock=mock_pixinsight)
    else:
        try:
            MultiscaleGradientCorrection(_calibrated_master()).save_to(
                out, mock=mock_pixinsight,
            )
        except PixInsightNoReferenceDataError:
            # MARS has no coverage for this field/filter. Fall back to the
            # data-driven AutomaticBackgroundExtractor (no external
            # reference), so gradient correction still works anywhere on sky.
            log.warning(
                "MGC has no MARS reference data; falling back to ABE",
                filter=stack.filter, master=str(master.filename),
            )
            AutomaticBackgroundExtractor(_calibrated_master()).save_to(
                out, mock=mock_pixinsight,
            )

    # Equalize each panel's sky background to a known pedestal so
    # adjacent panels match at the seam. WBPP's `ImageIntegration`
    # additive-with-scaling normalization references each integration
    # to its own reference frame, so two panels with identical physics
    # land at different sky baselines (~11 σ apart on synthetic data
    # before pedestaling, ~0.1 σ after). MGC removes intra-panel
    # gradients but leaves the cross-panel offset; PMM's photometric
    # scaling addresses the multiplicative part but can't flatten an
    # additive baseline difference. Applied here so every downstream
    # consumer of the post-processed master sees an equalized panel.
    # See `core/stacks/post_processing/background_pedestal.py`.
    if not mock_pixinsight:
        from stacks.post_processing.background_pedestal import (
            apply_background_pedestal_to_xisf,
        )
        apply_background_pedestal_to_xisf(
            Path(out), target=config.background_pedestal,
        )

    return Produced(out)
