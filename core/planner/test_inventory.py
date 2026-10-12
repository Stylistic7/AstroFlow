"""The value model: tiers read off a measured visibility curve, seams graded
by their step. Pure — no context, no masters on disk."""
import pytest

from planner.inventory import (
    EQUAL_TOLERANCE, MAX_WEIGHT, MIN_GAIN, SEAM_WEIGHT, STAR_COLOUR_H, VISIBLE_GAIN_UNIT, cap_tiers,
    curve_tiers, project_kind, seam_tier, star_colour_cap,
)

# A 4-hour master: at SNR 3 the field goes from 40% to 50% detected by 4x the
# integration (so the object is half the field); the higher levels climb too.
# The planner sums the climbs over the levels and divides by the object.
CURVE = {"integration_s": 4 * 3600,
         "visibility": {"thresholds": [3, 10, 30], "factors": [1.0, 1.25, 1.5, 2.0, 3.0, 4.0],
                        "fraction": [[0.40, 0.42, 0.435, 0.455, 0.475, 0.50],
                                     [0.20, 0.21, 0.22, 0.235, 0.25, 0.255],
                                     [0.05, 0.06, 0.065, 0.07, 0.075, 0.08]]}}
OBJECT = 0.50
STEP_GAINS = [0.04, 0.03, 0.04, 0.04, 0.035]        # summed over the three levels, per step
STEP_HOURS = [1.0, 1.0, 2.0, 4.0, 4.0]


def test_tiers_follow_the_curve_and_fall_off():
    tiers = curve_tiers(CURVE, hours_now=4.0)
    hours = [h for h, _ in tiers]
    values = [v for _, v in tiers]
    assert hours == pytest.approx(STEP_HOURS)
    assert values == pytest.approx([min(MAX_WEIGHT, g / OBJECT / h / VISIBLE_GAIN_UNIT)
                                    for g, h in zip(STEP_GAINS, STEP_HOURS)])
    assert values == sorted(values, reverse=True)                   # diminishing returns


def test_hours_already_shot_consume_the_first_tiers():
    tiers = curve_tiers(CURVE, hours_now=6.5)                       # into the 1.5x-2x tier
    assert tiers[0] == pytest.approx([1.5, STEP_GAINS[2] / OBJECT / STEP_HOURS[2] / VISIBLE_GAIN_UNIT])
    assert curve_tiers(CURVE, hours_now=16.0) == []                 # past 4x: no more hours would improve it


def test_a_flat_tail_stops_at_min_gain_and_a_steep_curve_is_capped():
    flat = {"integration_s": 4 * 3600,
            "visibility": {"thresholds": [3], "factors": [1.0, 2.0, 4.0], "fraction": [[0.40, 0.41, 0.4101]]}}
    tiers = curve_tiers(flat, hours_now=4.0)
    assert len(tiers) == 1 and tiers[0][1] * VISIBLE_GAIN_UNIT >= MIN_GAIN
    steep = {"integration_s": 3600, "visibility": {"thresholds": [3], "factors": [1.0, 2.0], "fraction": [[0.1, 0.5]]}}
    assert curve_tiers(steep, hours_now=1.0)[0][1] == MAX_WEIGHT


def test_no_curve_means_no_opinion():
    assert curve_tiers(None, 3.0) is None
    assert curve_tiers({"integration_s": 3600}, 3.0) is None
    v1 = {"integration_s": 3600, "visibility": {"factors": [1.0, 2.0], "fraction": [0.1, 0.5]}}
    assert curve_tiers(v1, 1.0) is None                              # one threshold: not yet re-analysed


def test_detected_everywhere_is_not_done():
    """An object fully detected at SNR 3 but weak at 30 still wants hours."""
    deep = {"integration_s": 10 * 3600,
            "visibility": {"thresholds": [3, 30], "factors": [1.0, 2.0, 4.0],
                           "fraction": [[0.50, 0.50, 0.50], [0.20, 0.35, 0.50]]}}
    tiers = curve_tiers(deep, hours_now=10.0)
    assert tiers and tiers[0][1] * VISIBLE_GAIN_UNIT == pytest.approx(0.15 / 0.50 / 10.0)


def test_a_small_object_is_judged_on_itself_not_the_empty_sky():
    """The same climbs on a nebula filling 10% of the field count as much as
    on one filling 50%: an hour improves the object, not the frame."""
    small = {"integration_s": 3600, "visibility": {"thresholds": [3, 30], "factors": [1.0, 2.0],
                                                    "fraction": [[0.08, 0.10], [0.02, 0.04]]}}
    big = {"integration_s": 3600, "visibility": {"thresholds": [3, 30], "factors": [1.0, 2.0],
                                                  "fraction": [[0.40, 0.50], [0.10, 0.20]]}}
    assert curve_tiers(small, 1.0)[0][1] == pytest.approx(curve_tiers(big, 1.0)[0][1])


def test_an_implausible_curve_is_not_trusted():
    bad = {"integration_s": 4 * 3600, "visibility": {"thresholds": [3], "factors": [1.0, 4.0], "fraction": [[0.80, 0.82]]}}
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
