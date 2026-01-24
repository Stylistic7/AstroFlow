from typing import Any

from pix.classes import ShellScriptTask, Image, ImageList
from pix.context import append_task
from pix.processes import enum, execute_global

_default_parameters = {
    "colorSpace": enum("LRGBCombination.prototype.RGB"),
    "channels": [],
    "inheritAstrometricSolution": True,
    "mL": 0.500,
    "mc": 0.500,
    "clipHighlights": True,
    "noiseReduction": False,
    "layersRemoved": 4,
    "layersProtected": 2,
}

_rename_javascript = """
var windows = ImageWindow.windows;
windows[windows.length - 1].currentView.id = "{view_id}";
"""


def LRGBCombination(
    images: ImageList, output_view_id: str = "LRGB", **kwargs: Any
) -> ImageList:
    # TODO: This is horrifically ugly.
    view_ids = [i.view_id for i in images]
    parameters = _default_parameters | kwargs
    channels = [f'[true, "{v}"]' for v in view_ids]
    parameters["channels"] = enum(f"[{', '.join(channels)}]")

    append_task(
        ShellScriptTask(javascript=execute_global("LRGBCombination", parameters))
    )

    append_task(
        ShellScriptTask(javascript=_rename_javascript.format(view_id=output_view_id))
    )

    return ImageList([Image(view_id=output_view_id, tag=output_view_id, dirty=True)])
