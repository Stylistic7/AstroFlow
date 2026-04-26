from suggestions.moon import recommend_filters

ALL = frozenset({"H", "S", "O", "L", "R", "G", "B"})
H_S_ONLY = frozenset({"H", "S"})
NARROWBAND = frozenset({"H", "S", "O"})


def test_moon_below_horizon_allows_all_filters():
    # Full moon below horizon is harmless.
    assert recommend_filters(illum=1.0, altitude_deg=-5.0) == ALL


def test_dim_moon_allows_all_filters():
    assert recommend_filters(illum=0.2, altitude_deg=60.0) == ALL


def test_bright_moon_up_restricts_to_h_and_s():
    # OIII (~500nm) is too close to moonlight; only H-alpha and SII clear.
    assert recommend_filters(illum=0.9, altitude_deg=40.0) == H_S_ONLY


def test_moderate_moon_up_allows_narrowband():
    assert recommend_filters(illum=0.5, altitude_deg=30.0) == NARROWBAND
