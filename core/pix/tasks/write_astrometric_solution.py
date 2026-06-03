"""WriteAstrometricSolution — inject a known WCS into a PixInsight view.

Use this when the image's astrometric solution has already been computed
externally (e.g., by astrometry.net's `solve-field`) and PixInsight's own
`ImageSolver` can't reach it (dense Milky Way fields after 8× downscale
where PI's pair-RANSAC star matching breaks while solve-field's
quad-hashing succeeds).

Mechanics: write the CD-matrix WCS keywords onto the view's FITS header,
then drive PI's own `ImageMetadata` (from WCSmetadata.jsh) to extract
them and emit the canonical PI representation — `PCL:AstrometricSolution:*`
XISF properties plus the linear-WCS keywords. SPCC, MultiscaleGradient-
Correction, and other downstream tools consume that property; they don't
care which solver produced it.

This is what `ImageSolver` itself does after it has a solution; we skip
the pair-RANSAC step that's failing and go directly to the persistence.
"""
import os
from pix.image import Image
from pix.recipe import Step


_SCRIPT_FILENAME = os.path.join(
    os.path.dirname(__file__), "write_astrometric_solution.js"
)


def WriteAstrometricSolution(image: Image, wcs_header: dict) -> Image:
    """Set a known astrometric solution on `image` from a CD-matrix WCS dict.

    `wcs_header` must contain at minimum CRVAL1, CRVAL2, CRPIX1, CRPIX2,
    CD1_1, CD1_2, CD2_1, CD2_2. Values are degrees / pixels per FITS
    convention. CTYPE is forced to RA---TAN/DEC--TAN — SIP polynomial
    coefficients are dropped (PI represents distortion through its own
    spline transformation, not SIP).
    """
    needed = ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
              "CD1_1", "CD1_2", "CD2_1", "CD2_2")
    missing = [k for k in needed if k not in wcs_header]
    if missing:
        raise ValueError(
            f"WriteAstrometricSolution requires {needed} in wcs_header; "
            f"missing: {missing}"
        )

    parameters = {
        "view_id": image.view_id,
        "crval1": float(wcs_header["CRVAL1"]),
        "crval2": float(wcs_header["CRVAL2"]),
        "crpix1": float(wcs_header["CRPIX1"]),
        "crpix2": float(wcs_header["CRPIX2"]),
        "cd1_1": float(wcs_header["CD1_1"]),
        "cd1_2": float(wcs_header["CD1_2"]),
        "cd2_1": float(wcs_header["CD2_1"]),
        "cd2_2": float(wcs_header["CD2_2"]),
    }
    image.recipe.add_step(Step(
        name="WriteAstrometricSolution",
        script_filename=_SCRIPT_FILENAME,
        parameters={k: repr(v) if isinstance(v, float) else str(v)
                    for k, v in parameters.items()},
    ))
    return image
