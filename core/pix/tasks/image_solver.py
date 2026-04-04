"""ImageSolver — astrometric plate solving in PixInsight.

ImageSolver is a script (not a built-in process), invoked via -p parameter
flags in the .scp file rather than inline JavaScript.
"""
import re
from typing import Any

from pix.image import Image
from pix.recipe import Step

_DEFAULT_PARAMETERS = {
    "metadata_focal": "541.5685232722849",
    "metadata_useFocal": "false",
    "metadata_xpixsz": "3.759999990463257",
    "metadata_resolution": "0.0003977929324021598",
    "metadata_referenceSystem": "ICRS",
    "metadata_ra": "311.71182622264297",
    "metadata_dec": "30.854210446991924",
    "metadata_epoch": "2460126.7458429397",
    "metadata_observationTime": "2460127.3486342593",
    "metadata_topocentric": "false",
    "solver_version": "6.3.1",
    "solver_magnitude": "12",
    "solver_autoMagnitude": "true",
    "solver_generateErrorImg": "false",
    "solver_structureLayers": "5",
    "solver_minStructureSize": "0",
    "solver_hotPixelFilterRadius": "1",
    "solver_noiseReductionFilterRadius": "0",
    "solver_sensitivity": "0.5",
    "solver_peakResponse": "0.5",
    "solver_brightThreshold": "3",
    "solver_maxStarDistortion": "0.6",
    "solver_autoPSF": "false",
    "solver_catalogMode": "3",
    "solver_catalog": "GaiaDR3_XPSD",
    "solver_vizierServer": "https://vizier.cds.unistra.fr/",
    "solver_showStars": "false",
    "solver_showStarMatches": "false",
    "solver_showSimplifiedSurfaces": "false",
    "solver_showDistortion": "false",
    "solver_generateDistortModel": "false",
    "solver_distortionCorrection": "true",
    "solver_rbfType": "101",
    "solver_maxSplinePoints": "4000",
    "solver_splineOrder": "2",
    "solver_splineSmoothing": "0.004999999888241291",
    "solver_enableSimplifier": "true",
    "solver_simplifierRejectFraction": "0.10000000149011612",
    "solver_outlierDetectionRadius": "160",
    "solver_outlierDetectionMinThreshold": "4",
    "solver_outlierDetectionSigma": "5",
    "solver_useActive": "true",
    "solver_outSuffix": "_ast",
    "solver_projection": "0",
    "solver_projectionOriginMode": "0",
    "solver_restrictToHQStars": "false",
    "solver_intersectionMode": "1",
    "solver_tryApparentCoordinates": "true",
    "solver_tryExhaustiveInitialAlignment": "false",
    "non_interactive": "true",
}

_SCRIPT_FILENAME = "/Applications/PixInsight/src/scripts/AdP/ImageSolver.js"

_ACTIVATE_JS = """
let view = View.viewById("{view_id}");
let window = view.window;
window.bringToFront();
"""


def ImageSolver(image: Image, **kwargs: Any) -> Image:
    parameters = _DEFAULT_PARAMETERS | {k: str(v) for k, v in kwargs.items()}

    # Bring the target view to the front before running the script
    image.recipe.add_step(Step(
        name="ImageSolver (activate)",
        javascript=_ACTIVATE_JS.format(view_id=image.view_id),
    ))

    # Run ImageSolver as an external script with -p parameter flags
    image.recipe.add_step(Step(
        name="ImageSolver",
        script_filename=_SCRIPT_FILENAME,
        parameters=parameters,
        success_pattern=re.compile(r'Resolution \.+ [\d.]+ arcsec/px'),
    ))

    return Image(
        view_id=image.view_id,
        recipe=image.recipe,
        filter=image.filter,
    )
