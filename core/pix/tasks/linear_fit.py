"""LinearFit — linear scaling of images to match a reference."""
from typing import Any

from pix.image import Image
from pix.recipe import Step
from pix.tasks.js import execute_on_views

_DEFAULT_PARAMETERS = {
    "referenceViewId": "",
    "rejectLow": 0.000000,
    "rejectHigh": 0.920000,
}


def LinearFit(images: list[Image], reference: Image, **kwargs: Any) -> list[Image]:
    """Scale images to match reference brightness.

    Returns the input images (modified in-place in PixInsight).
    """
    view_ids = [i.view_id for i in images]
    parameters = _DEFAULT_PARAMETERS | kwargs | {"referenceViewId": reference.view_id}

    recipe = images[0].recipe
    js = execute_on_views("LinearFit", parameters, view_ids)
    recipe.add_step(Step(javascript=js, name="LinearFit"))

    return [
        Image(view_id=img.view_id, recipe=recipe, filter=img.filter)
        for img in images
    ]
