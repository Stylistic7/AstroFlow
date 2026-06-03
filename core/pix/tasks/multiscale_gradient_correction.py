"""MultiscaleGradientCorrection — gradient correction using MARS database."""

import re
from typing import Any

from pix.image import Image
from pix.recipe import Step
from pix.tasks.js import execute_on_views

_DEFAULT_PARAMETERS = {
    "useMARSDatabase": True,
    "marsDatabaseFiles": [
        [True, "/Volumes/Astro/Catalogs/MARS-DR1-1.1.1.xmars"]
    ],
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


def MultiscaleGradientCorrection(image: Image, **kwargs: Any) -> Image:
    from pix.tasks.filters import normalize_filter

    filter_key = normalize_filter(image.filter) or "Ha"
    mars_filter = _MARS_FILTER_MAP.get(filter_key, "Ha")

    js = execute_on_views(
        "MultiscaleGradientCorrection",
        _DEFAULT_PARAMETERS | {"grayMARSFilter": mars_filter} | kwargs,
        [image.view_id],
    )
    image.recipe.add_step(
        Step(
            javascript=js,
            name="MultiscaleGradientCorrection",
            error_patterns=[
                re.compile(r"\*\*\* Error:.*lacks flux calibration metadata"),
                re.compile("No reference data found for filter"),
            ],
        )
    )
    return Image(
        view_id=image.view_id,
        recipe=image.recipe,
        filter=image.filter,
    )
