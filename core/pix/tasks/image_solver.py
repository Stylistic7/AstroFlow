import os
from typing import Any

from pix.classes import ShellScriptTask, FilenameList, ImageList
from pix.context import append_task

# TODO: These parameters must be floats, not strings, so they can be overriden
# by the user.
_default_parameters = {
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
    "solver_autoMagnitude": "false",
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
    "solver_catalogMode": "1",
    "solver_vizierServer": "https://vizier.cds.unistra.fr/",
    "solver_showStars": "false",
    "solver_showStarMatches": "false",
    "solver_showSimplifiedSurfaces": "false",
    "solver_showDistortion": "false",
    "solver_generateDistortModel": "false",
    "solver_catalog": "PPMXL",
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

_activate_javascript = """
let view = View.viewById("{view_id}");
let window = view.window;
window.bringToFront();
"""


def ImageSolver(images: ImageList, **kwargs: Any) -> ImageList:
    print("### ImageSolver() images are", images)
    parameters = _default_parameters | kwargs
    javascript_filename = "/Applications/PixInsight/src/scripts/AdP/ImageSolver.js"

    # TODO: The `javascript_filename` thing is long. Maybe there can be two
    # ShellScriptTask types, one that accepts code, and one that accepts a filename.
    for image in images:
        append_task(
            ShellScriptTask(
                javascript=_activate_javascript.format(view_id=image.view_id)
            )
        )
        append_task(
            ShellScriptTask(
                javascript_filename=javascript_filename, parameters=parameters
            )
        )

    return images
