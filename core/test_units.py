"""Unit-definition regression pins for the photometric gauge.

The first draft's conversions were caught INVERTED on review
(2026-09-01) — the algebra is one division away from silently wrong, so
pin it: calibration makes pixel = SPFC_REFERENCE_SCALE × physical.
"""
import math

from units import (
    SPFC_REFERENCE_SCALE,
    Degrees,
    Radians,
    SpfcBandFlux,
    band_flux_from_pixel,
)


def test_gauge_definition_direction():
    # A source of 1000 SPFC band-flux units on a calibrated master:
    physical = SpfcBandFlux(1000.0)
    assert math.isclose(physical.to_pixel_value(),
                        1000.0 * SPFC_REFERENCE_SCALE)
    # ... and 1.0 pixel is a BIG physical flux (1/REF), not a tiny one.
    assert math.isclose(band_flux_from_pixel(1.0),
                        1.0 / SPFC_REFERENCE_SCALE)


def test_gauge_roundtrip():
    v = 0.005  # the sky pedestal
    assert math.isclose(band_flux_from_pixel(v).to_pixel_value(), v,
                        rel_tol=1e-12)


def test_reference_scale_frozen():
    # Redefining the unit re-keys every calibrated master — the value is
    # part of the gauge DEFINITION, not a tunable.
    assert SPFC_REFERENCE_SCALE == 6.0e-3


def test_angle_units_still_sane():
    assert math.isclose(Degrees(180.0).to_radians(), math.pi)
    assert math.isclose(Radians(math.pi).to_degrees(), 180.0)
