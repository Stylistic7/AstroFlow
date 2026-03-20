"""BlurXTerminator — AI-based deconvolution."""
from typing import Any

from pix2.image import Image
from pix2.recipe import Step
from pix2.tasks.js import execute_on_views

_DEFAULT_PARAMETERS = {
    "ai_file": "BlurXTerminator.4.mlpackage",
    "correct_only": False,
    "correct_first": False,
    "nonstellar_then_stellar": False,
    "lum_only": False,
    "sharpen_stars": 0.25,
    "adjust_halos": 0.00,
    "nonstellar_psf_diameter": 0.00,
    "auto_nonstellar_psf": True,
    "sharpen_nonstellar": 0.90,
}


def BlurXTerminator(image: Image, **kwargs: Any) -> Image:
    js = execute_on_views(
        "BlurXTerminator",
        _DEFAULT_PARAMETERS | kwargs,
        [image.view_id],
    )
    image.recipe.add_step(Step(javascript=js, name="BlurXTerminator"))
    return Image(
        view_id=image.view_id,
        recipe=image.recipe,
        filter=image.filter,
    )
