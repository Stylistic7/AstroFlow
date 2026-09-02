"""Synthetic multi-panel narrowband mosaics with Cygnus-calibrated errors.

One shared truth sky per channel (pedestal + channel-correlated nebulosity
+ real Gaia stars with photon-weighted narrowband fluxes), cropped into
overlapping panels, each corrupted by the error model MEASURED on the
Cygnus 24-panel mosaic:

  * gain: WBPP sibling masters differ 14% MAD/median, 1.79x worst pair
    (2026-08-20, triple-reproduced) — multiplicative, rides the signal.
  * sky offset: per-panel constants 0.8-1.3e-4 above the 0.005 pedestal
    (narrowband, measured 2026-08-31 by the overlap-constants solve).
  * noise: per-panel sigma varying ~2x (the only uncorrectable term).

Everything is linear float in production units (pedestal 0.005, stars to
~1.0). Panels come in two shapes: bbox-on-disk placed panels for
`_blend_many`, and WCS-stamped FITS for per-panel SPFC.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

PEDESTAL = 0.005

# Measured Cygnus error scales.
GAIN_SPREAD = (0.87, 1.55)       # spans the 1.79x worst pair
OFFSET_RANGE = (0.8e-4, 1.3e-4)  # narrowband per-panel sky constants
# Photometric noise: per-panel stack depth expressed as detected
# electrons at the 0.005 pedestal, plus a signal-independent read floor
# in linear units. Shot noise is signal-DEPENDENT: brighter pixels are
# noisier in electrons, quieter in relative terms. SNR scales as
# sqrt(electrons): a ~2.5x SNR spread across panels. The floor stays OUT
# of photon starvation (a 100 e- panel cannot pass SPFC's star-quality
# filters, and production REFUSES such panels rather than including
# known-bad data — Stylistic7, 2026-09-01; the refusal path is exercised by
# post_stack_calibrate failing loudly, not by the standing labs).
# Floor calibrated empirically: on 1200x800 lab panels (~24 catalog
# stars <= G14), SPFC clears its 5-sample bar from ~550 e- up and fails
# at <= 286 (4 samples at 254 e-, run 19). Real >=30-min narrowband
# stacks sit above this floor.
# Units rescaled 2026-09-01 (x6.52) alongside the labs' star gauge
# (x0.392) so lab masters carry the PRODUCTION flux scale: SPFC
# measures ~6e-3 (the frozen _SPFC_REFERENCE_SCALE) and applied gains
# land ~1.0, exactly like real masters. Star SNR ~ k*sqrt(e) is
# invariant under the paired rescale, so SPFC sample counts, refusal
# margins, and draw marginality are preserved EXACTLY.
# Solve-starvation history (runs 27-31, 2026-09-01): panel 15's Hα
# looked unsolvable at this depth, but a x4 depth raise did NOT fix it —
# the real blocker was smooth bright nebulosity skewing solve-field's
# extraction (267 stars once flattened), now handled by the flatten
# retry rung in plate_solving.solve_field. The x4 was reverted the same
# day: the rung makes it unnecessary, and it changed nothing else —
# S/O truth-r measured identical (0.93/0.90) at both depths; that drop
# was the metric's stale star-mask threshold (fixed in _truth_r), not
# the scene. Do not "fix" solve failures with depth.
E_AT_PEDESTAL_RANGE = (3260.0, 10430.0)
READ_SIGMA_RANGE = (1e-5, 4e-5)


@dataclass(frozen=True)
class PanelSpec:
    """One panel: canvas crop + its injected corruption."""
    row0: int
    col0: int
    height: int
    width: int
    gain: float
    offset: float
    electrons_per_unit: float    # detected e- per 1.0 of linear signal
    read_sigma: float            # signal-independent noise, linear units


def make_panel_specs(canvas_h: int, canvas_w: int, grid: tuple[int, int],
                     overlap_frac: float, rng) -> list[PanelSpec]:
    """A grid of overlapping panels tiling the canvas, with per-panel
    corruption drawn from the measured Cygnus ranges. The first panel is
    pinned to gain 1.0 / offset 0 so injected values are recoverable as
    ratios/differences against it."""
    rows, cols = grid
    ph = int(canvas_h / (rows - (rows - 1) * overlap_frac))
    pw = int(canvas_w / (cols - (cols - 1) * overlap_frac))
    step_r = (canvas_h - ph) // max(rows - 1, 1)
    step_c = (canvas_w - pw) // max(cols - 1, 1)
    specs = []
    for r in range(rows):
        for c in range(cols):
            first = r == 0 and c == 0
            e_at_ped = float(rng.uniform(*E_AT_PEDESTAL_RANGE))
            specs.append(PanelSpec(
                row0=r * step_r, col0=c * step_c, height=ph, width=pw,
                gain=1.0 if first else float(rng.uniform(*GAIN_SPREAD)),
                offset=0.0 if first else float(rng.uniform(*OFFSET_RANGE)),
                electrons_per_unit=e_at_ped / PEDESTAL,
                read_sigma=float(rng.uniform(*READ_SIGMA_RANGE)),
            ))
    return specs


def make_truth_skies(canvas_h: int, canvas_w: int, star_layers: dict,
                     rng) -> dict:
    """Per-channel truth: pedestal + channel-correlated nebulosity + stars.

    H carries the base structure; S and O mix part of H's structure with
    their own (real SHO fields are correlated but not identical), at the
    lower surface brightness of those lines.
    """
    from scipy.ndimage import gaussian_filter

    def cloud(scale_px, amplitude):
        field = gaussian_filter(rng.normal(0, 1, (canvas_h, canvas_w)),
                                scale_px)
        field -= field.min()
        return field / max(field.max(), 1e-12) * amplitude

    base = cloud(80, 1.0)
    mixes = {"H": (1.0, 0.020), "S": (0.5, 0.008), "O": (0.3, 0.006)}
    skies = {}
    for ch, (corr, amp) in mixes.items():
        own = cloud(60, 1.0)
        structure = corr * base + (1 - corr) * own
        structure = structure / max(structure.max(), 1e-12) * amp
        skies[ch] = PEDESTAL + structure + star_layers[ch]
    return skies


def corrupt_panel(truth: np.ndarray, spec: PanelSpec, rng) -> np.ndarray:
    """What WBPP would hand the mosaic pipeline for this panel: the true
    sky seen through this night's transparency (gain), sky glow (offset),
    and this stack's depth — photometric noise with a shot term whose
    variance tracks the expected signal in detected electrons (gain and
    sky glow both carry photons, so both are inside the Poisson expectation)
    plus a signal-independent read floor."""
    crop = truth[spec.row0:spec.row0 + spec.height,
                 spec.col0:spec.col0 + spec.width]
    expected = crop * spec.gain + spec.offset
    electrons = np.clip(expected * spec.electrons_per_unit, 0.0, None)
    shot = rng.normal(0.0, 1.0, expected.shape) * np.sqrt(electrons) \
        / spec.electrons_per_unit
    read = rng.normal(0.0, spec.read_sigma, expected.shape)
    return np.clip(expected + shot + read, 1e-6, None).astype(np.float32)


def write_placed_panel_file(path: Path, panel: np.ndarray, spec: PanelSpec,
                            canvas_h: int, canvas_w: int) -> Path:
    """Persist as the bbox-on-disk placed-panel format `_blend_many` reads."""
    from mosaics.placement.panel_io import write_placed_panel

    write_placed_panel(
        path, panel.astype(np.float32),
        offset_xy=(spec.col0, spec.row0),
        canvas_size=(canvas_w, canvas_h),
    )
    return path


def panel_wcs_header(spec: PanelSpec, canvas_wcs: dict) -> dict:
    """The panel's own TAN WCS: the canvas solution with CRPIX shifted to
    the crop origin. Diagonal CD only — the parity/shear conventions that
    bit the round-trip lab (see write_astrometric_solution) never arise."""
    return {
        "CRVAL1": canvas_wcs["CRVAL1"],
        "CRVAL2": canvas_wcs["CRVAL2"],
        "CRPIX1": canvas_wcs["CRPIX1"] - spec.col0,
        "CRPIX2": canvas_wcs["CRPIX2"] - spec.row0,
        "CD1_1": canvas_wcs["CD1_1"], "CD1_2": 0.0,
        "CD2_1": 0.0, "CD2_2": canvas_wcs["CD2_2"],
    }


def write_panel_fits(path: Path, panel: np.ndarray, filter_key: str,
                     wcs: dict) -> Path:
    """A master-like float FITS for per-panel SPFC: pixels + FILTER +
    DATE-OBS, NO baked WCS cards (PI must see only the injected solution
    — round-trip lab lesson).

    Rows are flipped on write: PixInsight loads FITS top-down while the
    renderer paints in astropy's bottom-up frame, so a hand-authored
    WCS only binds to the stars if the stored rows are inverted
    (verified 2026-09-01: same panel, same solution — 0 SPFC samples
    as-is, 63 samples flipped)."""
    from astropy.io import fits

    hdu = fits.PrimaryHDU(data=np.flipud(panel).astype(np.float32))
    hdu.header["FILTER"] = filter_key
    hdu.header["DATE-OBS"] = "2026-04-18T06:00:00.000"
    hdu.header["EXPOSURE"] = 300.0
    hdu.header["INSTRUME"] = "ZWO ASI6200MM Pro"
    hdu.writeto(path, overwrite=True)
    return path
