"""MultiscaleGradientCorrection — gradient correction using MARS database."""

import html
import re
from pathlib import Path
from typing import Any

from pix.image import Image
from pix.recipe import Step
from pix.tasks.js import execute_on_views

_DEFAULT_PARAMETERS = {
    "useMARSDatabase": True,
    "marsDatabaseFiles": None,  # [[True, <each database registered in PixInsight>]]
    "grayMARSFilter": "Ha",
    "referenceImageId": "",
    "gradientScale": 1024,
    "structureSeparation": 3,
    "modelSmoothness": 1.00,
    "minFieldRatio": 0.017,
    "maxFieldRatio": 0.167,
    "enforceFieldLimits": True,
    "showGradientModel": True,
}

_MARS_FILTER_MAP = {
    "L": "L",
    "R": "R",
    "G": "G",
    "B": "B",
    "H": "Ha",
    "O": "OIII",
    "S": "SII",
}


# Where PixInsight keeps each instance's settings, the registered MARS
# databases among them.
_PI_SETTINGS = Path.home() / "Library" / "PixInsight"


def registered_mars_databases() -> list[str]:
    """The MARS databases registered in PixInsight on this Mac (MGC's MARS
    database dialog), in its order. A script-run MGC doesn't use them by
    itself: with none named it stops at "No MARS database files have been
    selected" (tested 2026-10-17), so they are read from PixInsight's own
    settings and named."""
    found = []
    for settings in sorted(_PI_SETTINGS.glob("core-*-pxi.settings")):
        for path in re.findall(r'<v k="MARSDatabaseFilePath\d+" t="s">([^<]*)</v>', settings.read_text(errors="ignore")):
            path = html.unescape(path)
            if path not in found and Path(path).is_file():
                found.append(path)
    return found


def MultiscaleGradientCorrection(image: Image, **kwargs: Any) -> Image:
    from pix.tasks.filters import normalize_filter

    databases = registered_mars_databases()
    if not databases:
        raise FileNotFoundError("MultiscaleGradientCorrection needs a MARS database: in PixInsight, open "
                                "MultiscaleGradientCorrection and add one under MARS databases")
    filter_key = normalize_filter(image.filter) or "Ha"
    mars_filter = _MARS_FILTER_MAP.get(filter_key, "Ha")

    js = execute_on_views(
        "MultiscaleGradientCorrection",
        _DEFAULT_PARAMETERS | {"marsDatabaseFiles": [[True, d] for d in databases],
                               "grayMARSFilter": mars_filter} | kwargs,
        [image.view_id],
    )
    image.recipe.add_step(
        Step(
            javascript=js,
            name="MultiscaleGradientCorrection",
            error_patterns=[
                re.compile(r"\*\*\* Error:.*lacks flux calibration metadata"),
            ],
            # "No reference data found for filter" (MARS has no coverage for
            # this field/filter) is detected universally by Recipe and
            # raised as PixInsightNoReferenceDataError, which
            # post_process_stack catches to fall back to ABE.
        )
    )
    return Image(
        view_id=image.view_id,
        recipe=image.recipe,
        filter=image.filter,
    )
