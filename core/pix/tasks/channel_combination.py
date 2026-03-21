"""ChannelCombination — combine multiple images into one color image."""
from typing import Any

from pix.image import Image
from pix.recipe import Step
from pix.tasks.js import Enum, execute_global

_DEFAULT_PARAMETERS = {
    "colorSpace": Enum("ChannelCombination.prototype.RGB"),
    "channels": [],
    "inheritAstrometricSolution": True,
}

_RENAME_JS = """
var windows = ImageWindow.windows;
windows[windows.length - 1].currentView.id = "{view_id}";
"""


def ChannelCombination(
    images: list[Image], output_view_id: str = "RGB", **kwargs: Any
) -> Image:
    view_ids = [i.view_id for i in images]
    parameters = _DEFAULT_PARAMETERS | kwargs
    channels = [f'[true, "{v}"]' for v in view_ids]
    parameters["channels"] = Enum(f"[{', '.join(channels)}]")

    recipe = images[0].recipe
    js = execute_global("ChannelCombination", parameters)
    js += "\n" + _RENAME_JS.format(view_id=output_view_id)

    recipe.add_step(Step(javascript=js, name="ChannelCombination"))
    return Image(view_id=output_view_id, recipe=recipe)
