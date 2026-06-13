"""CatalogStarGenerator — render Gaia DR3 stars at an image's WCS.

Wraps the Andrés del Pozo / PTeam script at
`/Applications/PixInsight/src/scripts/CatalogStarGenerator/CatalogStarGenerator.js`
v2.2.2 (PJSR 2.0 / V8; the script carries its own `#engine v8`). Used by
the calibration-validation harness to give synthetic light frames real
catalog stars — without them WBPP's SubframeSelector fails to measure
every frame and the integration step never runs.

The engine registers parameter fields under prefix ``sgen_`` (see
`CatalogStarGeneratorEngine.js` `__base__( CSG_SETTINGS_MODULE, "sgen",
… )` and WCSmetadata.jsh `MakeParamsKey` which builds
``<prefix>_<field>``). The ``non_interactive`` flag is read unprefixed
by the script's ``main()`` and selects the no-GUI path that targets
``ImageWindow.activeWindow``.

Catalog selection uses the display name registered in
``__catalogRegister__``; for local Gaia DR3 XPSD that is
``"Gaia DR3 (XPSD)"`` (AstronomicalCatalogs).
"""
from __future__ import annotations

import re
from typing import Any

from pix.image import Image
from pix.recipe import Step

_SCRIPT_FILENAME = (
    "/Applications/PixInsight/src/scripts/CatalogStarGenerator/CatalogStarGenerator.js"
)

# Prefix from CSG_SETTINGS_MODULE base call. Every registered field on
# the engine object lives at "sgen_<field>" in the Parameters store.
_P = "sgen_"

_DEFAULT_PARAMETERS: dict[str, str] = {
    # WCS source: read from the active view's astrometric solution.
    f"{_P}useImageMetadata": "true",
    # Catalog: the local Gaia DR3 XPSD database the user has under
    # /Volumes/Astro/Catalogs/, registered as "Gaia DR3 (XPSD)" per
    # AstronomicalCatalogs.jsh.
    f"{_P}catalogName": "Gaia DR3 (XPSD)",
    # Star magnitude bounds. Defaults chosen to bracket "detectable to
    # PMM" stars on a small field while still rendering enough faint
    # catalog sources for SPCC's photometric matching.
    f"{_P}minMagnitude": "8.0",
    f"{_P}maxMagnitude": "16.0",
    # backgroundMagnitude high (= dim) so the output image is dominated
    # by the rendered stars, not a flat sky pedestal. The physics-model
    # harness adds its own sky/bias/dark/flat on top.
    f"{_P}backgroundMagnitude": "25.0",
    # PSF model. 0 = Gauss, 1 = Moffat. Moffat with beta ≈ 4 gives
    # realistic seeing-limited wings that PMM's star detector can fit.
    f"{_P}PSF": "1",
    f"{_P}beta": "4.0",
    # FWHM in arcsec (per CSG source line 1479
    # `fwhm_pix = this.FWHM / ( this.metadata.resolution * 3600 )`).
    f"{_P}FWHM": "3.0",
    f"{_P}generateStars": "true",
    f"{_P}generateNoise": "false",
    # main() reads this unprefixed; selects the no-GUI execution path
    # that uses ImageWindow.activeWindow as the target.
    "non_interactive": "true",
}


# Bring the target view to the front so `ImageWindow.activeWindow`
# resolves to it inside the script. Same shape as image_solver.py uses.
_ACTIVATE_JS = """
let view = View.viewById("{view_id}");
let window = view.window;
window.bringToFront();
"""


def CatalogStarGenerator(image: Image, **kwargs: Any) -> Image:
    """Render Gaia DR3 stars at `image`'s WCS into a new view.

    Returns an Image whose pixel data contains catalog stars rendered
    as Moffat PSFs at sub-pixel WCS-projected positions. The returned
    view ID is ``<image.view_id>_stars`` so adjacent stars-only frames
    don't collide.

    Any kwarg overrides the corresponding ``sgen_*`` default (e.g.
    ``maxMagnitude=14`` becomes ``sgen_maxMagnitude=14``). Pass
    ``non_interactive`` or other unprefixed parameters by setting them
    explicitly; we don't add a prefix to keys the caller supplies.
    """
    output_view_id = f"{image.view_id}_stars"

    parameters = _DEFAULT_PARAMETERS | {
        f"{_P}outputImageId": output_view_id,
    }
    # Prefix only kwargs whose names match a registered field; pass
    # `non_interactive` and any other unprefixed keys through as-is.
    for k, v in kwargs.items():
        if k == "non_interactive" or k.startswith(_P):
            parameters[k] = str(v)
        else:
            parameters[f"{_P}{k}"] = str(v)

    image.recipe.add_step(Step(
        name="CatalogStarGenerator (activate)",
        javascript=_ACTIVATE_JS.format(view_id=image.view_id),
    ))

    image.recipe.add_step(Step(
        name="CatalogStarGenerator",
        script_filename=_SCRIPT_FILENAME,
        parameters=parameters,
        # CSG's Generate() throws "The target image has no astrometric
        # solution" if WCS extraction fails. Catch it loudly — silently
        # producing a star-less output would let WBPP run on bad input.
        error_patterns=[
            re.compile(r"The target image has no astrometric solution"),
            re.compile(r"\*\*\* Error:"),
        ],
    ))

    return Image(
        view_id=output_view_id,
        recipe=image.recipe,
        filter=image.filter,
    )
