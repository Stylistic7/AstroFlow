"""The value model: tiers read off a measured visibility curve, seams graded
by their step. Pure — no context, no masters on disk."""
import pytest

from planner.inventory import (
    EQUAL_TOLERANCE, MAX_WEIGHT, MIN_GAIN, SEAM_WEIGHT, STAR_COLOUR_H, VISIBLE_GAIN_UNIT, cap_tiers,
    curve_tiers, project_kind, seam_tier, star_colour_cap,
)

# A 4-hour master whose field goes from 40% visible to 57% at 4x the integration.
CURVE = {"integration_s": 4 * 3600,
         "visibility": {"factors": [1.0, 1.25, 1.5, 2.0, 3.0, 4.0],
                        "fraction": [0.40, 0.44, 0.47, 0.51, 0.55, 0.57]}}


def test_tiers_follow_the_curve_and_fall_off():
    tiers = curve_tiers(CURVE, hours_now=4.0)
    hours = [h for h, _ in tiers]
    values = [v for _, v in tiers]
    assert hours == pytest.approx([1.0, 1.0, 2.0, 4.0, 4.0])
    assert values == pytest.approx([0.04 / VISIBLE_GAIN_UNIT, 0.03 / VISIBLE_GAIN_UNIT, 0.02 / VISIBLE_GAIN_UNIT,
                                    0.01 / VISIBLE_GAIN_UNIT, 0.005 / VISIBLE_GAIN_UNIT])
    assert values == sorted(values, reverse=True)                   # diminishing returns


def test_hours_already_shot_consume_the_first_tiers():
    tiers = curve_tiers(CURVE, hours_now=6.5)                       # into the 1.5x-2x tier
    assert tiers[0] == pytest.approx([1.5, 0.02 / VISIBLE_GAIN_UNIT])
    assert curve_tiers(CURVE, hours_now=16.0) == []                 # past 4x: no more hours would improve it


def test_a_flat_tail_stops_at_min_gain_and_a_steep_curve_is_capped():
    flat = {"integration_s": 4 * 3600,
            "visibility": {"factors": [1.0, 2.0, 4.0], "fraction": [0.40, 0.41, 0.4101]}}
    tiers = curve_tiers(flat, hours_now=4.0)
    assert len(tiers) == 1 and tiers[0][1] * VISIBLE_GAIN_UNIT >= MIN_GAIN
    steep = {"integration_s": 3600, "visibility": {"factors": [1.0, 2.0], "fraction": [0.1, 0.5]}}
    assert curve_tiers(steep, hours_now=1.0)[0][1] == MAX_WEIGHT


def test_no_curve_means_no_opinion():
    assert curve_tiers(None, 3.0) is None
    assert curve_tiers({"integration_s": 3600}, 3.0) is None


def test_an_implausible_curve_is_not_trusted():
    bad = {"integration_s": 4 * 3600, "visibility": {"factors": [1.0, 4.0], "fraction": [0.80, 0.82]}}
    assert curve_tiers(bad, 4.0) is None                            # the Siril-pedestal failure


def test_seam_hours_close_the_step_and_value_grows_with_it():
    assert seam_tier(1.0, 1.0, 5.0) is None                          # within tolerance
    hours, value = seam_tier(1.5, 1.0, 4.0)
    assert hours == pytest.approx(4.0 * ((1.5 / EQUAL_TOLERANCE) ** 2 - 1))
    assert value == pytest.approx(SEAM_WEIGHT * 0.5)
    assert seam_tier(2.0, 1.0, 4.0)[1] > value


def test_project_kind_from_the_hours_shot():
    assert project_kind({"H": 12.0, "O": 8.0, "R": 1.0, "G": 1.0, "B": 1.0}) == "narrowband"
    assert project_kind({"L": 6.0, "R": 2.0, "G": 2.0, "B": 2.0, "H": 3.0}) == "lrgb"    # a galaxy with some Hα
    assert project_kind({}) == "lrgb"


def test_star_colour_is_capped_only_on_narrowband_projects():
    assert star_colour_cap("B", "narrowband", 0.1) == pytest.approx(STAR_COLOUR_H - 0.1)
    assert star_colour_cap("B", "narrowband", 1.0) == 0.0                  # 20 min is plenty
    assert star_colour_cap("B", "lrgb", 1.0) is None                      # a galaxy: the curve decides
    assert star_colour_cap("H", "narrowband", 1.0) is None
    assert cap_tiers([[1.0, 2.0], [3.0, 1.0]], 2.5) == [[1.0, 2.0], [1.5, 1.0]]
    assert cap_tiers([[1.0, 2.0]], 0.0) == []
