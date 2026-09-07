"""WriteAstrometricSolution — inject a known WCS into a PixInsight view.

Use this when the image's astrometric solution has already been computed
externally (e.g., by astrometry.net's `solve-field`) and PixInsight's own
`ImageSolver` can't reach it (dense Milky Way fields after 8× downscale
where PI's pair-RANSAC star matching breaks while solve-field's
quad-hashing succeeds).

Mechanics: discard whatever solution the view already carries (PI's FITS
loader builds one from keywords embedded in the file — the ASIAir writes
its own, less accurate WCS into every sub), hand PI's `ImageMetadata`
(WCSmetadata.jsh) our solution as `PCL:AstrometricSolution:*` properties,
and let it emit the canonical representation — the full property set plus
the linear-WCS keywords. SPFC, SPCC, MGC and other downstream tools consume
that property set; they don't care which solver produced it.

Properties, not keywords, because a keyword-sourced solution goes through
PI's bottom-up guess (the sign of a CDELT2 it estimates from the CD
matrix's rotation branch), which for det>0 matrices mirrors the solution
for half the camera orientations — every +90° frame paired 70% of its
stars with the wrong Gaia entries (2026-09-07). A property-sourced
solution is taken verbatim.
"""
import os
import re
from pix.image import Image
from pix.recipe import Step


_SCRIPT_FILENAME = os.path.join(
    os.path.dirname(__file__), "write_astrometric_solution.js"
)


def WriteAstrometricSolution(image: Image, wcs_header: dict) -> Image:
    """Set a known astrometric solution on `image` from a CD-matrix WCS dict.

    `wcs_header` must contain at minimum CRVAL1, CRVAL2, CRPIX1, CRPIX2,
    CD1_1, CD1_2, CD2_1, CD2_2. Values are degrees / pixels per FITS
    convention, on the image's rows in file order. CTYPE is forced to
    RA---TAN/DEC--TAN — SIP polynomial coefficients are dropped (PI
    represents distortion through its own spline transformation, not SIP;
    the loss is ~0.3 px median, 1 px max on a full 6200MM frame).
    """
    needed = ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
              "CD1_1", "CD1_2", "CD2_1", "CD2_2")
    missing = [k for k in needed if k not in wcs_header]
    if missing:
        raise ValueError(
            f"WriteAstrometricSolution requires {needed} in wcs_header; "
            f"missing: {missing}"
        )

    parameters = {"view_id": image.view_id}
    parameters.update({k.lower(): float(wcs_header[k]) for k in needed})
    image.recipe.add_step(Step(
        name="WriteAstrometricSolution",
        script_filename=_SCRIPT_FILENAME,
        parameters={k: repr(v) if isinstance(v, float) else str(v)
                    for k, v in parameters.items()},
        # A JS abort (e.g. a PJSR API misuse) leaves the loader's solution
        # in place and the rest of the recipe runs on it — silently, since
        # the universal `*** Error:` catch-all is exempted for recipes with
        # a success-pattern step. Require our own completion line.
        success_pattern=re.compile(r"WriteAstrometricSolution: done\."),
    ))
    return image
