from typing import Any, cast

from pix.classes import ShellScriptTask, ImageList
from pix.context import append_task
from pix.processes import enum, execute_on_view_ids

_default_parameters = {
    "zoomFactor": -2,
    "downsamplingMode": enum("IntegerResample.prototype.Average"),
    "xResolution": 72.000,
    "yResolution": 72.000,
    "metric": False,
    "forceResolution": False,
    "gammaCorrection": False,
    "noGUIMessages": True,
}


def IntegerResample(images: ImageList, **kwargs: Any) -> ImageList:
    assert all(i.view_id is not None for i in images)
    view_ids = [cast(str, i.view_id) for i in images]

    append_task(
        ShellScriptTask(
            javascript=execute_on_view_ids(
                "IntegerResample", _default_parameters | kwargs, view_ids
            )
        )
    )

    return images
