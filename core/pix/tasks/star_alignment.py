"""StarAlignment — register images to a reference frame."""
from typing import Any

from pix.image import Image
from pix.recipe import Step
from pix.tasks.js import Enum, execute_global

_DEFAULT_PARAMETERS = {
    "structureLayers": 5,
    "sensitivity": 0.50,
    "peakResponse": 0.50,
    "brightThreshold": 3.00,
    "maxStarDistortion": 0.60,
    "distortionCorrection": True,
    "localDistortion": True,
    "localDistortionScale": 256,
    "matcherTolerance": 0.0500,
    "ransacTolerance": 2.00,
    "referenceImage": "",
    "referenceIsFile": False,
    "targets": [],
    "mode": Enum("StarAlignment.RegisterMatch"),
    "writeKeywords": True,
    "generateMasks": False,
    "generateDrizzleData": True,
    "inheritAstrometricSolution": True,
    "noGUIMessages": True,
    "useSurfaceSplines": True,
    "pixelInterpolation": Enum("StarAlignment.Auto"),
    "clampingThreshold": 0.30,
    "onError": Enum("StarAlignment.Continue"),
    "fitPSF": Enum("StarAlignment.FitPSF_DistortionOnly"),
    "intersection": Enum("StarAlignment.MosaicOnly"),
}

_RENAME_JS = 'View.viewById("{old_id}").id = "{new_id}";'


def StarAlignment(images: list[Image], **kwargs: Any) -> list[Image]:
    """Align images to the first image in the list.

    Returns new Images with '_aligned' suffix view IDs.
    """
    if len(images) < 2:
        return images

    reference = images[0]
    targets = images[1:]
    recipe = reference.recipe

    target_entries = [f'[true, false, "{img.view_id}"]' for img in targets]
    parameters = _DEFAULT_PARAMETERS | kwargs | {
        "referenceImage": reference.view_id,
        "targets": Enum(f"[{', '.join(target_entries)}]"),
    }

    js = execute_global("StarAlignment", parameters)

    # Normalize output view ids to the "_aligned" suffix this wrapper returns.
    # StarAlignment (verified on PI 1.9.4) leaves the reference view untouched
    # and names each registered target "<id>_registered" — NOT "_r" (that is
    # the file-output postfix) and NOT "_aligned". So rename the reference, and
    # rename each registered target from "_registered" to "_aligned".
    renames = [(reference.view_id, reference.view_id + "_aligned")]
    renames += [(t.view_id + "_registered", t.view_id + "_aligned") for t in targets]
    for old_id, new_id in renames:
        js += "\n" + _RENAME_JS.format(old_id=old_id, new_id=new_id)

    recipe.add_step(Step(javascript=js, name="StarAlignment"))

    return [
        Image(
            view_id=img.view_id + "_aligned",
            recipe=recipe,
            filter=img.filter,
        )
        for img in images
    ]
