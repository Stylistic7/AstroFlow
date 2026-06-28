"""HDRComposition — combine images of varying exposures into one HDR master.

PixInsight's HDRComposition replaces saturated regions of the longer
exposure with the shorter exposure scaled by the exposure ratio, then
feathers the seam. When no saturation is present it degrades to a
weighted blend, so it covers both intentional-HDR and accidental-mixed
exposure cases on the PixInsight path.

HDRComposition is a FILE-based process: its `images` parameter is a list of
`[enabled, path]` entries naming files on DISK, which it opens itself — it
does NOT operate on open views (verified against the installed PixInsight
1.9.4 process: passing view ids yields `*** Error: FileFormat: Empty format
name…` and `executeGlobal()` returns false). So we write each (already
registered) input view to a temp file and compose those files. Inputs must
be on the same pixel grid; pre-register them with `StarAlignment`.
"""
from typing import Any

from pix.image import Image
from pix.recipe import Step
from pix.tasks.js import Enum, format_value


# Exactly the parameters HDRComposition exposes in PixInsight 1.9.4 — verified
# against `(new HDRComposition).toSource()` on the installed process. There is
# NO `combinationMethod` parameter and NO `HDRComposition.HDRC_MaximumValue`
# enum (both read `undefined` on a fresh instance), and NO `outputHints`;
# setting a non-existent process parameter is what broke the n>1 HDR path.
# `outputMasks=False` keeps HDRComposition from opening extra mask windows, so
# the only window it creates is the HDR result (auto-named "HDR").
_DEFAULT_PARAMETERS = {
    "images": [],
    "inputHints": "",
    "maskBinarizingThreshold": 0.80,
    "maskSmoothness": 8,
    "maskGrowth": 2,
    "replaceLargeScales": 0,
    "autoExposures": True,
    "rejectBlack": True,
    "useFittingRegion": False,
    "fittingRectX0": 0,
    "fittingRectY0": 0,
    "fittingRectX1": 0,
    "fittingRectY1": 0,
    "generate64BitResult": True,
    "outputMasks": False,
    "closePreviousImages": False,
}


def HDRComposition(
    images: list[Image],
    output_view_id: str = "HDR",
    **kwargs: Any,
) -> Image:
    """Compose multiple registered exposures into one HDR image.

    `images` must be sorted longest-exposure-first (HDRComposition treats the
    first as the long-exposure base and replaces its saturated regions with
    subsequent shorter exposures) and already registered to a common pixel
    grid (e.g. by `StarAlignment`).

    Because HDRComposition reads FILES, each input view is first written to a
    temp file in the recipe's work dir; those paths are then composed. The id
    HDRComposition gives its output window is context-dependent and unreliable,
    so we never assume it: we snapshot the open windows before `executeGlobal()`
    and rename whatever NEW window appears to ``output_view_id`` (with
    `outputMasks=False` the composed result is the only new window).
    `executeGlobal()` is checked — a false return (e.g. exposure estimation or
    registration failed) raises immediately with a clear message.
    """
    if len(images) == 1:
        return images[0]

    recipe = images[0].recipe

    # HDRComposition opens files, not views — dump each registered input view
    # to a temp file, then compose those file paths.
    dump_lines: list[str] = []
    image_entries: list[str] = []
    for i, img in enumerate(images):
        tmp_path = recipe.tmp_dir / f"_hdrc_input_{i}.xisf"
        dump_lines.append(
            f'View.viewById("{img.view_id}").window.saveAs('
            f'"{tmp_path}", false, false, true, false);'
        )
        image_entries.append(f'[true, "{tmp_path}"]')

    parameters = _DEFAULT_PARAMETERS | kwargs | {
        "images": Enum(f"[{', '.join(image_entries)}]"),
    }

    lines = list(dump_lines)
    # Record existing windows so the new (composed) window can be identified.
    lines.append("var _hdrcBefore = {};")
    lines.append("for (var _i = 0; _i < ImageWindow.windows.length; ++_i)")
    lines.append("   _hdrcBefore[ImageWindow.windows[_i].mainView.id] = true;")
    lines.append("var P = new HDRComposition;")
    for key, value in parameters.items():
        lines.append(f"P.{key} = {format_value(value)};")
    lines.append(
        'if (!P.executeGlobal())\n'
        '   throw new Error("HDRComposition.executeGlobal() returned false — '
        'check the PixInsight console (exposure estimation or registration may '
        'have failed).");'
    )
    # Find the one new window HDRComposition created and give it our id. Only
    # rename if it differs — assigning a view its OWN id makes PixInsight treat
    # the id as already-in-use and bump it (HDR -> HDR1), which would then make
    # the downstream `viewById(output_view_id)` save fail.
    lines.append("var _hdrcOut = null;")
    lines.append("for (var _i = 0; _i < ImageWindow.windows.length; ++_i)")
    lines.append("   if (!_hdrcBefore[ImageWindow.windows[_i].mainView.id]) {")
    lines.append("      _hdrcOut = ImageWindow.windows[_i]; break; }")
    lines.append(
        'if (_hdrcOut === null)\n'
        '   throw new Error("HDRComposition produced no output window.");'
    )
    lines.append(f'if (_hdrcOut.mainView.id != "{output_view_id}")')
    lines.append(f'   _hdrcOut.mainView.id = "{output_view_id}";')

    recipe.add_step(Step(javascript="\n".join(lines), name="HDRComposition"))
    return Image(view_id=output_view_id, recipe=recipe)
