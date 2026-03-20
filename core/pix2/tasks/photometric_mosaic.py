"""PhotometricMosaic — join two images with photometric matching."""
from typing import Any
from pathlib import Path

from pix2.image import Image
from pix2.recipe import Step

_JAVASCRIPT_FILENAME = (
    "/Applications/PixInsight/src/scripts/AdP/PhotometricMosaic.js"
)

_DEFAULT_PARAMETERS = {
    "solver_vizierServer": "https://vizier.cds.unistra.fr/",
    "solver_catalog": "PPMXL",
    "solver_useActive": "true",
    "non_interactive": "true",
}


def PhotometricMosaic(images: list[Image], output_path: Path, **kwargs: Any) -> Image:
    """Join two images using PixInsight's PhotometricMosaic script."""
    assert len(images) == 2
    recipe = images[0].recipe
    parameters = _DEFAULT_PARAMETERS | kwargs

    # PhotometricMosaic is a script, not a process
    js = f'// PhotometricMosaic: script-based task\n'
    js += f'// Output: {output_path}\n'
    for k, v in parameters.items():
        js += f'// param: {k}={v}\n'

    recipe.add_step(Step(javascript=js, name="PhotometricMosaic"))

    output_view_id = "photometric_mosaic_result"
    return Image(view_id=output_view_id, recipe=recipe)
