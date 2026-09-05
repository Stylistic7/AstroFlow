import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from catalogs.ngc2000_catalog import _all_entries, find_best_object
from catalogs.position_string import canonical_target_string
from geometry.distance import distance
from units import Degrees


def test_distance_requires_degrees():
    with pytest.raises(BeartypeCallHintParamViolation):
        distance(1.0, 2.0, 3.0, 4.0)


def test_distance_known_values():
    assert distance(Degrees(0), Degrees(0), Degrees(180), Degrees(0)) == pytest.approx(180.0)
    assert distance(Degrees(10), Degrees(0), Degrees(11), Degrees(0)) == pytest.approx(1.0)
    # At dec 60, one degree of RA spans half a degree of sky.
    assert distance(Degrees(10), Degrees(60), Degrees(11), Degrees(60)) == pytest.approx(
        0.5, abs=0.01
    )


def test_negative_declination_parse():
    # NGC1976 (M42) is listed at dec "-05 27": the sign applies to the
    # arcminutes too (the old parser added them, yielding -4.55).
    m42 = next(e for e in _all_entries() if e["id"] == "1976")
    assert m42["dec_degrees"] == pytest.approx(-(5 + 27 / 60))


def test_vizier_catalogs_loaded():
    entries = _all_entries()
    counts = {}
    for e in entries:
        prefix = str(e["id"]).split("-")[0].split(" ")[0]
        counts[prefix] = counts.get(prefix, 0) + 1
    assert counts["Sh2"] == 313
    assert counts["LBN"] == 1125
    assert counts["LDN"] == 1791
    assert counts["vdB"] == 158
    assert sum(v for k, v in counts.items() if k.startswith("B")) == 349

    sh2_132 = next(e for e in entries if e["id"] == "Sh2-132")
    assert sh2_132["name"] == "Lion Nebula"
    assert sh2_132["named"]
    assert sh2_132["ra_degrees"] == pytest.approx(334.75, abs=0.3)
    assert sh2_132["dec_degrees"] == pytest.approx(56.2, abs=0.3)
    assert sh2_132["radius"] == pytest.approx(0.75)


def test_naming_well_known_fields():
    cases = [
        ((334.70, 56.14), "Lion Nebula"),  # Sh2-132
        ((10.68, 41.27), "Andromeda Galaxy"),
        ((85.25, -2.46), "Horsehead Nebula"),  # B33, not Barnard's Loop
        ((315.7, 44.31), "North America Nebula"),
        ((324.75, 57.6), "Elephant's Trunk Nebula"),
        ((38.2, 61.45), "Heart Nebula"),
        ((344.28, 62.62), "Cave Nebula"),  # Sh2-155
        ((85.43, -1.85), "Flame Nebula"),  # inside Barnard's Loop
        ((303.06, 38.35), "Crescent Nebula"),  # inside gamma Cyg region
    ]
    for (ra, dec), expected in cases:
        _, entry = find_best_object(Degrees(ra), Degrees(dec))
        assert entry["name"] == expected, f"({ra}, {dec}) -> {entry['name']}"


def test_find_best_object_far_from_everything():
    # A position with no sized object nearby falls back to nearest-by-distance.
    d, entry = find_best_object(Degrees(40.0), Degrees(-35.0))
    assert entry is not None
    assert np.isfinite(d)


def test_canonical_target_string_accepts_plain_floats():
    assert canonical_target_string([334.70, 56.14]) == "Lion Nebula"
