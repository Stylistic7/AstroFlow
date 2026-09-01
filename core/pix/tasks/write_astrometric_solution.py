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


def _shear_free_cd(cd1_1: float, cd1_2: float,
                   cd2_1: float, cd2_2: float,
                   probe_px: float = 2000.0) -> tuple:
    """Project the CD matrix onto its nearest rotation × diagonal-scale.

    PI's WCSmetadata.jsh converts CD to legacy CDELT/CROTA by estimating
    the rotation TWICE (from cd2_1's and cd1_2's sign branches) and
    averaging. Solve-field matrices carry ~1e-7 deg/px shear, so near 0°
    or 180° the two off-diagonals can land on the same side of zero, the
    two estimates come out ~0° and ~180°, and the average is 90° —
    collapsing the scale and mirroring the regenerated solution (found
    2026-09-01: SPFC cross-matched 0 stars on such frames). A shear-free
    matrix provably keeps both estimates in consistent branches, and the
    discarded shear is sub-pixel across a full frame.

    Polar decomposition CD = R·P (R carries any reflection, so parity is
    preserved); drop P's off-diagonal. The discarded shear moves a point
    `probe_px` from the reference by well under a pixel for real
    solve-field output (~0.5 px/kpx measured); the guard falls back to
    the original matrix only for genuinely deformed input (≥3 px).
    """
    import numpy as np

    cd = np.array([[cd1_1, cd1_2], [cd2_1, cd2_2]], float)
    u, s, vt = np.linalg.svd(cd)
    r = u @ vt
    p = r.T @ cd
    clean = r @ np.diag(np.diag(p))

    corners = np.array([[probe_px, 0], [0, probe_px], [probe_px, probe_px]]).T
    err_px = np.abs((clean - cd) @ corners).max() / np.abs(np.diag(p)).min()
    if not np.isfinite(err_px) or err_px >= 3.0:
        return cd1_1, cd1_2, cd2_1, cd2_2
    return tuple(float(v) for v in clean.ravel())


def WriteAstrometricSolution(image: Image, wcs_header: dict) -> Image:
    """Set a known astrometric solution on `image` from a CD-matrix WCS dict.

    `wcs_header` must contain at minimum CRVAL1, CRVAL2, CRPIX1, CRPIX2,
    CD1_1, CD1_2, CD2_1, CD2_2. Values are degrees / pixels per FITS
    convention. CTYPE is forced to RA---TAN/DEC--TAN — SIP polynomial
    coefficients are dropped (PI represents distortion through its own
    spline transformation, not SIP), and the CD matrix is projected to
    its nearest rotation × scale (see `_shear_free_cd`) so PI's legacy
    CDELT/CROTA reconstruction cannot mirror the solution.
    """
    needed = ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
              "CD1_1", "CD1_2", "CD2_1", "CD2_2")
    missing = [k for k in needed if k not in wcs_header]
    if missing:
        raise ValueError(
            f"WriteAstrometricSolution requires {needed} in wcs_header; "
            f"missing: {missing}"
        )

    cd1_1, cd1_2, cd2_1, cd2_2 = _shear_free_cd(
        float(wcs_header["CD1_1"]), float(wcs_header["CD1_2"]),
        float(wcs_header["CD2_1"]), float(wcs_header["CD2_2"]),
    )
    parameters = {
        "view_id": image.view_id,
        "crval1": float(wcs_header["CRVAL1"]),
        "crval2": float(wcs_header["CRVAL2"]),
        "crpix1": float(wcs_header["CRPIX1"]),
        "crpix2": float(wcs_header["CRPIX2"]),
        "cd1_1": cd1_1,
        "cd1_2": cd1_2,
        "cd2_1": cd2_1,
        "cd2_2": cd2_2,
    }
    image.recipe.add_step(Step(
        name="WriteAstrometricSolution",
        script_filename=_SCRIPT_FILENAME,
        parameters={k: repr(v) if isinstance(v, float) else str(v)
                    for k, v in parameters.items()},
    ))
    return image
