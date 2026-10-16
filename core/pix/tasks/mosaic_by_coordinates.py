"""MosaicByCoordinates — register images to a common coordinate frame."""
from typing import Any

from pix.image import Image
from pix.recipe import Step
from third_party_tools import pixinsight

def _script_filename() -> str:
    return pixinsight("src", "scripts", "MosaicByCoordinates", "MosaicByCoordinates.js")

_DEFAULT_PARAMETERS = {
    "engine_centerCoordsAuto": True,
    "engine_resolutionAuto": True,
    "engine_rotationAuto": True,
    "engine_projectionAuto": False,
    "engine_projection": 3,
    "engine_dimensionsAuto": True,
    "engine_qualityHQ": True,
    "engine_pixelInterpolation": 5,
    "engine_clampingThreshold": 0.3,
    "engine_suffix": '"_registered"',
    "engine_overwrite": False,
    "engine_errorPolicy": 1,
    "non_interactive": True,
}


def MosaicByCoordinates(images: list[Image], **kwargs: Any) -> list[Image]:
    """Register images to a common coordinate frame using plate solutions."""
    recipe = images[0].recipe
    parameters = _DEFAULT_PARAMETERS | kwargs

    # MosaicByCoordinates expects filenames, not view IDs
    filenames = [img.view_id for img in images]  # TODO: use actual filenames
    parameters["engine_files"] = "|".join(filenames)

    recipe.add_step(Step(
        name="MosaicByCoordinates",
        script_filename=_script_filename(),
        parameters={k: str(v) for k, v in parameters.items()},
    ))

    return [
        Image(
            view_id=img.view_id + "_registered",
            recipe=recipe,
            filter=img.filter,
        )
        for img in images
    ]
