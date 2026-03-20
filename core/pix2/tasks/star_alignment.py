"""StarAlignment — register images to a reference frame."""
from typing import Any

from pix2.image import Image
from pix2.recipe import Step
from pix2.tasks.js import execute_on_views

_DEFAULT_PARAMETERS = {
    "referenceImage": "",
    "referenceIsFile": False,
    "writeKeywords": True,
    "generateMasks": False,
    "generateDrizzleData": False,
    "useSurfaceSplines": True,
    "sensitivity": 0.50,
    "noGUIMessages": True,
    "onError": 1,  # Continue
}

_RENAME_JS = """
var windows = ImageWindow.windows;
for (var i = windows.length - 1; i >= 0; i--) {{
    if (windows[i].currentView.id.indexOf("{suffix}") !== -1) {{
        windows[i].currentView.id = "{new_id}";
        break;
    }}
}}
"""


def StarAlignment(images: list[Image], **kwargs: Any) -> list[Image]:
    """Align images to the first image in the list.

    Returns new Images with '_aligned' suffix view IDs.
    """
    if len(images) < 2:
        raise ValueError("StarAlignment requires at least 2 images")

    reference = images[0]
    targets = images[1:]
    recipe = reference.recipe

    parameters = _DEFAULT_PARAMETERS | kwargs | {
        "referenceImage": reference.view_id,
        "referenceIsFile": False,
    }

    target_view_ids = [img.view_id for img in targets]
    js = execute_on_views("StarAlignment", parameters, target_view_ids)

    # Rename aligned outputs
    for img in targets:
        aligned_id = img.view_id + "_aligned"
        js += "\n" + _RENAME_JS.format(suffix="_registered", new_id=aligned_id)

    recipe.add_step(Step(javascript=js, name="StarAlignment"))

    result = [reference]
    for img in targets:
        result.append(Image(
            view_id=img.view_id + "_aligned",
            recipe=recipe,
            filter=img.filter,
        ))
    return result
