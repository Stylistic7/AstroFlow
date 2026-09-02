"""Read-through cache for the PI Gaia DR3/SP cone search (testdata).

The query is DETERMINISTIC — a pure function of the search params over
the static DR3/SP database — yet costs ~10 min of PixInsight per lab
run and was the flakiest stage of every run (2026-09-01: one hot-spin
wedge and one mid-query death, both in this stage). Stylistic7's call:
cache it as testdata. A miss runs the live query once and writes an
.npz keyed by the params; every later run loads in <1 s with no
PixInsight involvement at all.

Spectra are stored float32: the source samples are 8-bit quantized on
the XP grid, so float32 is lossless; they are cast back to float64 on
load so cached and live results are numerically interchangeable
downstream.
"""
from pathlib import Path

import numpy as np

TESTDATA_DIR = Path(__file__).parent / "testdata"


def _cache_key(ra, dec, radius, mag_high, mag_low, source_limit) -> str:
    return (f"gaia_xpsd_ra{ra:g}_dec{dec:g}_r{radius:.4f}"
            f"_mag{mag_low:g}to{mag_high:g}_n{source_limit}")


def cached_xpsd_stars(
    center_ra_deg: float,
    center_dec_deg: float,
    radius_deg: float,
    magnitude_high: float,
    magnitude_low: float = -1.5,
    source_limit: int = 20_000,
) -> dict:
    """Drop-in for `pix.tasks.gaia_xpsd_query.query_xpsd_stars`."""
    path = TESTDATA_DIR / (_cache_key(
        center_ra_deg, center_dec_deg, radius_deg,
        magnitude_high, magnitude_low, source_limit) + ".npz")
    if path.exists():
        z = np.load(path)
        return {
            "wavelength_nm": z["wavelength_nm"].astype(np.float64),
            "stars": [
                (float(ra), float(dec), float(g),
                 spec.astype(np.float64))
                for ra, dec, g, spec in zip(
                    z["ra"], z["dec"], z["g"], z["spectra"])
            ],
        }

    from pix.tasks.gaia_xpsd_query import query_xpsd_stars
    cat = query_xpsd_stars(center_ra_deg, center_dec_deg, radius_deg,
                           magnitude_high, magnitude_low, source_limit)
    save_catalog(path, cat)
    return cat


def save_catalog(path: Path, cat: dict) -> None:
    TESTDATA_DIR.mkdir(exist_ok=True)
    np.savez_compressed(
        path,
        wavelength_nm=np.asarray(cat["wavelength_nm"]),
        ra=np.array([s[0] for s in cat["stars"]]),
        dec=np.array([s[1] for s in cat["stars"]]),
        g=np.array([s[2] for s in cat["stars"]]),
        spectra=np.stack(
            [s[3] for s in cat["stars"]]).astype(np.float32),
    )
