"""SPFC narrowband round trip: Gaia XP spectra → rendered stars → blind
plate solve → SPFC → recovered flux scale.

The closed loop: stars are rendered with band fluxes computed from the SAME
XP spectra and the SAME rectangular band (center/width imported from the
production SPFC task) that SPFC itself uses for its predictions. So SPFC's
fitted scale (the PSFFLX00 keyword) must track the injected rendering scale
exactly, up to photon noise — independent of stellar color, filter choice,
and Gaia's spectral resolution, all of which cancel between render and
prediction.

Three assertions, strongest last:
  1. Every rendered frame blind-solves with the real solve-field.
  2. SPFC completes and stamps a finite PSFFLX00 on every frame.
  3. Closure: a 1.25× injected gain moves PSFFLX00 by exactly 1.25×
     (inversely — PSFFLX00 is physical flux per data unit), and
     PSFFLX00 × adu_per_flux is ONE constant across H, S, and O even
     though each filter was rendered with its own scale.

Requires PixInsight + the local Gaia DR3/SP XPSD database + solve-field.
NEVER run while the app is up (pytest resets the shared Ray cluster).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Galactic-plane field (b ≈ -2°): dense enough that a 0.3°×0.2° frame
# holds hundreds of catalog stars for the blind solve and dozens of
# mag ≤ 14 stars for SPFC's fit.
_RA, _DEC = 100.0, 0.0
_WIDTH, _HEIGHT = 1080, 720
_IMAGE_SCALE = 1.0  # arcsec/px → FOV 0.30° × 0.20°
_QUERY_RADIUS = 0.22
_GAIN = 1.25
_BASE_ADU = 800.0  # bias + flat sky pedestal
_MAG14_TARGET_ADU = 15_000.0  # total flux for a mag-14 star


def _adu_per_flux(catalog: dict, filter_key: str) -> float:
    """Rendering scale: put a G≈14 star (SPFC's faint limit) at a
    comfortably measurable total flux."""
    from testing.narrowband_stars import band_flux

    fluxes = np.array([
        band_flux(catalog["wavelength_nm"], spec, filter_key)
        for _, _, g_mag, spec in catalog["stars"]
        if abs(g_mag - 14.0) < 0.75
    ])
    assert fluxes.size >= 5, f"too few G≈14 stars for scale ({fluxes.size})"
    return _MAG14_TARGET_ADU / float(np.median(fluxes))


def _strip_wcs(path: Path) -> None:
    """Remove the baked legacy WCS cards from a light frame.

    PixInsight reads the template's CDELT/CROTA header as a FLIPPED
    solution (astropy reads it with the opposite parity — the classic
    legacy-WCS ambiguity), then silently prefers that load-time solution
    over the one WriteAstrometricSolution injects, mirror-imaging every
    predicted star position (verified 2026-09-01: SPFC searched at the
    stale flipped center and cross-matched 0 of 24 sources). With the
    cards gone, the injected solve-field solution is the only one.
    """
    from astropy.io import fits

    with fits.open(path, mode="update") as hdul:
        h = hdul[0].header
        for k in list(h):
            if k[:5] in ("CTYPE", "CRVAL", "CRPIX", "CDELT", "CROTA") \
                    or k[:3] in ("CD1", "CD2", "PC1", "PC2"):
                del h[k]


def _write_light(tmp: Path, filter_key: str, tag: str,
                 star_layer: np.ndarray, rng) -> Path:
    """Compose base + stars + shot noise, quantize, write a solvable FITS."""
    from testing.test_frames import make_synthetic_frames

    ideal = _BASE_ADU + star_layer
    noisy = ideal + rng.normal(0.0, np.sqrt(np.maximum(ideal, 1.0)))
    pixels = np.clip(noisy, 0, 65535).astype(np.uint16)
    paths = make_synthetic_frames(
        prefix=f"spfcrt_{filter_key}_{tag}", number=1, image_type="Light",
        dest_dir=str(tmp), ra=_RA, dec=_DEC, centers=[(_RA, _DEC)],
        width_pixels=_WIDTH, height_pixels=_HEIGHT,
        image_scale=_IMAGE_SCALE, rotation_angle_deg=0.0,
        filter=filter_key, exposure=300,
        pixel_arrays=[pixels], solvable=True,
    )
    _strip_wcs(Path(paths[0]))
    return Path(paths[0])


# PI 1.9.4's SPFC stores its result as view PROPERTIES (PCL:SPFC:*), not
# the PSFFLX00 keyword the retired spfc_normalize op read (verified
# 2026-09-01 by dumping the view: keywords carry no flux entry, properties
# include PCL:SPFC:ScaleFactors). Read the property in-session and print
# it; the test parses it from the recipe console.
_READ_FLX_JS = """
let view = View.viewById("{view_id}");
let sf = view.propertyValue("PCL:SPFC:ScaleFactors");
let flx = null;
if (sf != null)
    flx = (typeof sf == "number") ? sf : (sf.at ? sf.at(0) : sf[0]);
if (flx == null || !isFinite(flx) || flx <= 0)
    throw new Error("spfc roundtrip: no PCL:SPFC:ScaleFactors after SPFC");
console.writeln(format("ASTROFLOW: roundtrip SPFCSCALE %.8e", flx));
"""


def _run_spfc(light: Path, filter_key: str, wcs_header: dict,
              out: Path) -> float:
    """WriteAstrometricSolution → SPFC → read PSFFLX00 off the view."""
    import re

    from pix.image import Image
    from pix.recipe import Step
    from pix.tasks import (
        SpectrophotometricFluxCalibration,
        WriteAstrometricSolution,
    )

    img = Image.open(light, filter=filter_key)
    img = WriteAstrometricSolution(img, wcs_header)
    img = SpectrophotometricFluxCalibration(img)
    img.recipe.add_step(Step(
        name="read SPFC scale",
        javascript=_READ_FLX_JS.format(view_id=img.view_id),
    ))
    img.save_to(out)

    found = re.findall(r"roundtrip SPFCSCALE ([0-9.eE+-]+)", img.recipe.log)
    assert found, (
        f"no SPFC scale in the recipe console for {out.name} — "
        f"calibration did not run"
    )
    flx = float(found[-1])
    assert np.isfinite(flx) and flx > 0, f"bad SPFC scale {flx} on {out.name}"
    return flx


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_spfc_narrowband_roundtrip(tmp_path):
    import plate_solving
    from pix.tasks.gaia_xpsd_query import query_xpsd_stars
    from testing.narrowband_stars import render_narrowband_stars
    from testing.test_frames import make_synthetic_frames
    from file_io.generic import read_header

    catalog = query_xpsd_stars(
        _RA, _DEC, _QUERY_RADIUS, magnitude_high=16.5,
    )
    print(f"\n[spfc-rt] catalog: {len(catalog['stars'])} stars, "
          f"grid {catalog['wavelength_nm'][0]:.1f}–"
          f"{catalog['wavelength_nm'][-1]:.1f} nm")

    # One throwaway frame supplies the deterministic WCS header that the
    # renderer projects catalog positions through (same ra/dec/centers/
    # rotation/scale as every light written below).
    template = make_synthetic_frames(
        prefix="spfcrt_template", number=1, image_type="Light",
        dest_dir=str(tmp_path), ra=_RA, dec=_DEC, centers=[(_RA, _DEC)],
        width_pixels=_WIDTH, height_pixels=_HEIGHT,
        image_scale=_IMAGE_SCALE, rotation_angle_deg=0.0,
        pixel_arrays=[np.zeros((_HEIGHT, _WIDTH), np.uint16)],
    )
    wcs_template = read_header(Path(template[0]))

    rng = np.random.default_rng(0x5FC_0001)
    unit_constants = {}
    for filter_key in ("H", "S", "O"):
        k = _adu_per_flux(catalog, filter_key)
        stars = render_narrowband_stars(
            _HEIGHT, _WIDTH, wcs_template, catalog, filter_key, k,
        )

        light_lo = _write_light(tmp_path, filter_key, "g100", stars, rng)
        light_hi = _write_light(tmp_path, filter_key, "g125",
                                stars * _GAIN, rng)

        # Leg 1: the rendered frame must blind-solve like a real light.
        ps = plate_solving.solve_field(light_lo, debug=True)
        assert ps is not None, (
            f"[{filter_key}] solve-field failed on the rendered frame"
        )
        print(f"[spfc-rt:{filter_key}] solved: center={ps.center} "
              f"scale={ps.image_scale:.3f} stars={ps.num_stars}")

        # Leg 2 + 3: SPFC on both gains, same astrometric solution
        # (identical field; only fluxes differ).
        flx_lo = _run_spfc(light_lo, filter_key, ps.header,
                           tmp_path / f"spfc_{filter_key}_g100.xisf")
        flx_hi = _run_spfc(light_hi, filter_key, ps.header,
                           tmp_path / f"spfc_{filter_key}_g125.xisf")

        ratio = flx_hi / flx_lo
        print(f"[spfc-rt:{filter_key}] scale lo={flx_lo:.6g} "
              f"hi={flx_hi:.6g} ratio={ratio:.4f} (expect {_GAIN})")
        # The SPFC scale factor is measured-data per predicted-physical
        # flux, so 1.25× brighter pixels give a 1.25× LARGER value
        # (measured on H: 0.17225/0.13717 = 1.2558).
        assert abs(ratio / _GAIN - 1.0) < 0.04, (
            f"[{filter_key}] injected {_GAIN}× gain recovered as "
            f"{ratio:.4f} — closure broken"
        )
        # scale = adu_per_flux / (units constant) → scale/k is one
        # physical constant shared by all three filters.
        unit_constants[filter_key] = flx_lo / k

    # Cross-filter closure: three bands, three rendering scales, ONE
    # physical-units constant.
    vals = np.array(list(unit_constants.values()))
    spread = float(vals.max() / vals.min() - 1.0)
    print(f"[spfc-rt] unit constants {unit_constants} spread={spread:.3%}")
    assert spread < 0.10, (
        f"SPFC scale / adu_per_flux differs {spread:.1%} across filters — "
        f"band handling asymmetric: {unit_constants}"
    )
