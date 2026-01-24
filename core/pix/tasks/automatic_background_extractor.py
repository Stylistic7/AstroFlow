from typing import Any, cast

from pix.classes import ShellScriptTask, ImageList, Image
from pix.context import append_task
from pix.processes import enum, execute_on_view_ids

_default_parameters = {
    "tolerance": 1.000,
    "deviation": 0.800,
    "unbalance": 1.800,
    "minBoxFraction": 0.050,
    "maxBackground": 1.0000,
    "minBackground": 0.0000,
    "useBrightnessLimits": False,
    "polyDegree": 4,
    "boxSize": 5,
    "boxSeparation": 5,
    "modelImageSampleFormat": enum("AutomaticBackgroundExtractor.prototype.f32"),
    "abeDownsample": 2.00,
    "writeSampleBoxes": False,
    "justTrySamples": False,
    "targetCorrection": enum("AutomaticBackgroundExtractor.prototype.Subtract"),
    "normalize": False,
    "discardModel": True,
    "replaceTarget": True,
    "correctedImageId": "",
    "correctedImageSampleFormat": enum(
        "AutomaticBackgroundExtractor.prototype.SameAsTarget"
    ),
    "verboseCoefficients": False,
    "compareModel": False,
    "compareFactor": 10.00,
}


def AutomaticBackgroundExtractor(images: ImageList, **kwargs: Any) -> ImageList:
    assert all(i.view_id is not None for i in images)
    view_ids = [cast(str, i.view_id) for i in images]

    append_task(
        ShellScriptTask(
            javascript=execute_on_view_ids(
                "AutomaticBackgroundExtractor", _default_parameters | kwargs, view_ids
            )
        )
    )

    images = [
        Image(filename=image.filename, tag="abe", view_id=image.view_id, dirty=True)
        for image in images
    ]

    return images
