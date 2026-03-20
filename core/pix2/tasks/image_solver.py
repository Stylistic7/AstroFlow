"""ImageSolver — astrometric plate solving in PixInsight."""
from typing import Any

from pix2.image import Image
from pix2.recipe import Step

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
    "solver_autoMagnitude": "false",
    "solver_structureLayers": "5",
    "solver_sensitivity": "0.5",
    "solver_peakResponse": "0.5",
    "solver_catalogMode": "1",
    "solver_catalog": "PPMXL",
    "solver_distortionCorrection": "true",
    "solver_useActive": "true",
    "non_interactive": "true",
}

_ACTIVATE_JS = """
let view = View.viewById("{view_id}");
let window = view.window;
window.bringToFront();
"""

_JAVASCRIPT_FILENAME = (
    "/Applications/PixInsight/src/scripts/AdP/ImageSolver.js"
)


def ImageSolver(image: Image, **kwargs: Any) -> Image:
    parameters = _DEFAULT_PARAMETERS | {
        k: str(v) for k, v in kwargs.items()
    }

    # ImageSolver is a script, not a process — it needs the view activated
    # and parameters passed via -p flags. We encode this in the JavaScript.
    activate_js = _ACTIVATE_JS.format(view_id=image.view_id)

    # Build parameter string for the script
    param_assignments = "\n".join(
        f'// param: {k}={v}' for k, v in parameters.items()
    )

    js = (
        f'{activate_js}\n'
        f'// ImageSolver parameters:\n'
        f'{param_assignments}\n'
        f'// ImageSolver will be invoked as an external script'
    )

    image.recipe.add_step(Step(javascript=js, name="ImageSolver"))
    return Image(
        view_id=image.view_id,
        recipe=image.recipe,
        filter=image.filter,
    )
