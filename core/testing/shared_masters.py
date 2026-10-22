"""Test fixtures for the shared-master feature: a solved, stamped master on
disk (a small FITS with a TAN WCS and the header stamps a produced master
carries) beside the sidecar its producer would have written."""
from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np

from metadata.classes import ImageMetadata
from plate_solving import PlateSolution
from units import ArcsecondsPerPixel, Degrees, Radians


def tan_wcs_header(ra_deg: float, dec_deg: float, scale_arcsec: float, nx: int, ny: int,
                   rotation_deg: float = 0.0) -> dict:
    import math
    cdelt = scale_arcsec / 3600.0
    c, s = math.cos(math.radians(rotation_deg)), math.sin(math.radians(rotation_deg))
    return {
        "NAXIS": 2, "NAXIS1": nx, "NAXIS2": ny,
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CRVAL1": ra_deg, "CRVAL2": dec_deg,
        "CRPIX1": nx / 2, "CRPIX2": ny / 2,
        "CD1_1": -cdelt * c, "CD1_2": cdelt * s, "CD2_1": cdelt * s, "CD2_2": cdelt * c,
    }


def solved_master_metadata(
    path: Path, *, ra: float = 84.0, dec: float = 34.0, scale: float = 1.0,
    nx: int = 400, ny: int = 300, filter: str = "H", call_hash: str = "c0ffee0000000001",
    stable_hash: str = "0ddba11000000001", stamp: str = "feed000000000001",
    date=datetime.datetime(2026, 10, 1, 4, 0, 0), observer: str | None = None,
    exptime: float = 300.0, rotation_deg: float = 0.0,
) -> ImageMetadata:
    """An ImageMetadata for a post-stack-calibrated master, as its producer's
    sidecar would restore it."""
    header = {
        "IMAGETYP": "master light", "FILTER": filter, "EXPTIME": exptime, "EXPOSURE": exptime,
        "DATE-OBS": date, "INSTRUME": "ZWO ASI6200MM Pro", "CCD-TEMP": -10.0,
        "CALLHASH": call_hash, "STBLHASH": stable_hash, "PIPEHASH": stamp,
        "NAXIS1": nx, "NAXIS2": ny, "NOISE00": 0.0006,
    }
    if observer:
        header["OBSERVER"] = observer
    import math
    solution = PlateSolution(
        header=tan_wcs_header(ra, dec, scale, nx, ny, rotation_deg), num_stars=500,
        center=(Degrees(ra), Degrees(dec)), image_scale=ArcsecondsPerPixel(scale),
        rotation_angle=Radians(math.radians(rotation_deg)),
    )
    return ImageMetadata(
        content_hash=call_hash, stable_hash=stable_hash, filename=Path(path), header=header,
        width_pixels=nx, height_pixels=ny, channels=1, dtype=np.dtype(np.float32),
        plate_solution=solution, master_integration_s=exptime * 10, noise_estimate=0.0006,
    )


def write_shared_master(path: Path, meta: ImageMetadata | None = None, **kw) -> ImageMetadata:
    """Write a small FITS master at `path` with `meta`'s header, and the
    sidecar beside it (image_metadata + plate_solution), stamped with the
    file's size as the producer's sidecar would be."""
    from astropy.io import fits
    from file_io.sidecar import set_section
    from metadata.sidecar_io import write_image_metadata
    from plate_solving import _plate_solution_to_obj

    meta = meta or solved_master_metadata(path, **kw)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(np.zeros((meta.height_pixels, meta.width_pixels), dtype=np.float32))
    for k, v in meta.header.items():
        hdu.header[k] = v.isoformat() if isinstance(v, datetime.datetime) else v
    hdu.header["DONTSOLV"] = True   # blank pixels: a re-measure must not launch a blind solve
    hdu.writeto(path, overwrite=True)
    write_image_metadata(meta)
    set_section(path, "plate_solution", _plate_solution_to_obj(meta.plate_solution))
    return meta


def describe_for_coadd(path: Path, *, fwhm_arcsec: float = 3.0, scatter: float = 0.04) -> None:
    """Give a test master what the share gate (metadata/share_gate.py) reads
    of a real one: a measured PSF and a Gaia flux calibration in its sidecar."""
    from file_io.sidecar import set_section
    set_section(path, "analysis", {"version": 3, "noise_px": 1e-4, "pixel_arcsec": 2.0,
                                   "psf": {"fwhm_px": fwhm_arcsec / 2.0, "fwhm_arcsec": fwhm_arcsec, "n_stars": 300}})
    set_section(path, "spfc_calibration", {"scale": 0.01, "n_stars": 2000, "sigma": scatter, "reference": 0.006})
