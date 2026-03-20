"""StarXTerminator — AI-based star removal."""
from typing import Any

from pix2.image import Image
from pix2.recipe import Step
from pix2.tasks.js import execute_on_views

_DEFAULT_PARAMETERS = {
    "ai_file": "/Applications/PixInsight/library/StarXTerminator.11.mlmodel",
    "stars": True,
    "unscreen": False,
    "overlap": 0.20,
}

_RENAME_JS = """
var windows = ImageWindow.windows;
windows[windows.length - 1].currentView.id = "{stars_view_id}";
"""


def StarXTerminator(image: Image, **kwargs: Any) -> tuple[Image, Image]:
    """Remove stars from an image.

    Returns:
        (stars, starless) — two Images from the same recipe.
    """
    stars_view_id = image.view_id + "_stars"

    js = execute_on_views(
        "StarXTerminator",
        _DEFAULT_PARAMETERS | kwargs,
        [image.view_id],
    )
    js += "\n" + _RENAME_JS.format(stars_view_id=stars_view_id)

    image.recipe.add_step(Step(javascript=js, name="StarXTerminator"))

    stars = Image(
        view_id=stars_view_id,
        recipe=image.recipe,
        filter=image.filter,
    )
    starless = Image(
        view_id=image.view_id,
        recipe=image.recipe,
        filter=image.filter,
    )
    return stars, starless
