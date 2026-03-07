from typing import Any, cast

from pix.classes import ShellScriptTask, ImageList
from pix.context import append_task
from pix.processes import enum, execute_on_view_ids

# TODO: Find a way to store / specify the MARS database files.
_default_parameters = {
    "useMARSDatabase": True,
    "marsDatabaseFiles": [
        [True, "/Volumes/Astro/Catalogs/MARS-DR1-1.1.1.xmars"]
    ],
    "grayMARSFilter": "Ha",
    "redMARSFilter": "R",
    "greenMARSFilter": "G",
    "blueMARSFilter": "B",
    "referenceImageId": "",
    "gradientScale": 1024,
    "structureSeparation": 3,
    "modelSmoothness": 1.00,
    "minFieldRatio": 0.017,
    "maxFieldRatio": 0.167,
    "enforceFieldLimits": True,
    "scaleFactorRK": 1.0000,
    "scaleFactorG": 1.0000,
    "scaleFactorB": 1.0000,
    "showGradientModel": True,
    "command": "",
}


def MultiscaleGradientCorrection(images: ImageList, **kwargs: Any) -> ImageList:
    assert all(i.view_id is not None for i in images)
    view_ids = [cast(str, i.view_id) for i in images]

    filters = set([cast(str, i.filter) for i in images])
    assert len(filters) == 1
    filter = next(iter(filters))

    mars_filter = {
        "L": "L",
        "R": "R",
        "G": "G",
        "B": "B",
        "H": "Ha",
        "O": "OIII",
        "S": "SII",
    }[filter]

    append_task(
        ShellScriptTask(
            javascript=execute_on_view_ids(
                "MultiscaleGradientCorrection",
                _default_parameters | {"grayMARSFilter": mars_filter} | kwargs,
                view_ids,
            )
        )
    )

    return images
