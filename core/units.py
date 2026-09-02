import math
import numpy as np
from beartype import beartype


class Degrees(float):
    """An angle in degrees."""

    def to_radians(self) -> "Radians":
        return Radians(math.radians(self))


class Radians(float):
    """An angle in radians."""

    def to_degrees(self) -> Degrees:
        return Degrees(math.degrees(self))


class ArcsecondsPerPixel(float):
    """An image scale in arcseconds per pixel."""

    def to_degrees_per_pixel(self) -> float:
        return self / 3600.0


# ---------------------------------------------------------------------------
# Photometric units. No invented unit names (decision 2026-09-01): the
# HST pattern — calibrated pixels are a DECLARED engineering unit, and a
# header keyword carries the conversion to an established unit.
#
# Calibrated master pixels = SPFC_REFERENCE_SCALE x (flux in PI-SPFC
# band-flux units). Declared on every calibrated master as
# PHOTREF=SPFC_REFERENCE_SCALE. Follow-up (Stylistic7, 2026-09-01: anchor
# to Gaia's own units, no maggies): per-band keywords giving genuine SI
# photon flux through the filter — photons s^-1 m^-2 — per 1.0 pixel
# value. The conversion chain is already measured: pixel -> PHOTREF ->
# PI-SPFC units -> (round-trip lab's cross-band unit constant, 0.5%)
# -> XP-integral units [W m^-2 nm^-1 basis] -> /(hc) -> photon flux.

# The frozen gauge: SPFC-measured scale (raw data units per PI-SPFC
# band-flux unit) that calibration normalizes every master to.
# CALIBRATED 2026-09-01 on three live masters (6.19e-3 / 5.28e-3 /
# 8.48e-3, N = 5.8k-10.3k stars each), pinned near their median so a
# typical real master is numerically unchanged and the 0.005 sky
# pedestal regime holds. FROZEN: changing it re-keys every calibrated
# master.
SPFC_REFERENCE_SCALE = 6.0e-3


class SpfcBandFlux(float):
    """Flux in the unit system of PI SpectrophotometricFluxCalibration's
    predictions (proportional to the source's Gaia XP spectrum integrated
    photon-weighted across the production filter rectangle; PI's internal
    normalization sets the absolute scale). Named for its provenance, not
    presented as a universal unit; convertible to SI photon flux
    (photons s^-1 m^-2 through the band) via the measured per-band
    constants (see module note)."""

    def to_pixel_value(self) -> float:
        """Calibrated-master pixel value: SPFC_REFERENCE_SCALE x flux."""
        return float(self * SPFC_REFERENCE_SCALE)


def band_flux_from_pixel(pixel_value: float) -> SpfcBandFlux:
    """Invert the calibration gauge: pixel / SPFC_REFERENCE_SCALE."""
    return SpfcBandFlux(pixel_value / SPFC_REFERENCE_SCALE)


class SpfcScale(float):
    """An SPFC-measured flux scale: raw data units per SpfcBandFlux unit
    (PI's PCL:SPFC:ScaleFactors). Dividing a master by
    `scale / SPFC_REFERENCE_SCALE` normalizes it to the declared gauge."""


class Celsius(float):
    """A temperature in degrees Celsius."""


class Seconds(float):
    """A duration in seconds."""


@beartype
def ra_to_hms(ra: float) -> str:
    hours, remainder = np.divmod(ra * 3600, 15 * 3600)
    minutes, remainder = np.divmod(remainder, 15 * 60)
    seconds = remainder / 15
    return f"{hours:.0f}h {minutes:.0f}m {seconds:.0f}s"


@beartype
def dec_to_dms(dec: float) -> str:
    degrees, remainder = np.divmod(dec * 3600, 3600)
    minutes, remainder = np.divmod(remainder, 60)
    seconds = remainder
    return f"{degrees:.0f}° {minutes:.0f}' {seconds:.0f}\""
