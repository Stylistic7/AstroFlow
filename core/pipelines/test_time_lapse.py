"""`pipelines/time_lapse.py`: the night a run belongs to, which runs a night
holds, which night frames the canvas, and the stars fallback."""
import datetime
from types import SimpleNamespace

import numpy as np
from frozendict import frozendict

from decisions.decisions import DecisionKey, DecisionOverride
from pipelines.time_lapse import (
    canvas_night, night_of, nights_of, nights_window, runs_through, stars_fallback,
)

UTC = datetime.timezone.utc


def _run(run_id, start):
    return SimpleNamespace(stable_hash=run_id, oldest_date=start, frames=("f",) * 3, exposure=300.0)


def test_a_night_is_the_evening_it_started_on():
    assert night_of(datetime.datetime(2024, 9, 30, 3, 0, tzinfo=UTC)) == datetime.date(2024, 9, 29)
    assert night_of(datetime.datetime(2024, 9, 29, 23, 0, tzinfo=UTC)) == datetime.date(2024, 9, 29)
    assert night_of(datetime.datetime(2024, 9, 30, 13, 0, tzinfo=UTC)) == datetime.date(2024, 9, 30)


def test_nights_and_the_runs_through_each():
    a = _run("a", datetime.datetime(2024, 9, 30, 4, 0, tzinfo=UTC))    # night 09-29
    b = _run("b", datetime.datetime(2024, 10, 8, 5, 0, tzinfo=UTC))    # night 10-07
    c = _run("c", datetime.datetime(2024, 10, 8, 6, 0, tzinfo=UTC))    # night 10-07
    stack = SimpleNamespace(contents={"a": a, "b": b, "c": c})
    assert nights_of([stack]) == [datetime.date(2024, 9, 29), datetime.date(2024, 10, 7)]
    assert runs_through(stack, datetime.date(2024, 9, 29), frozenset()) == ("a",)
    assert runs_through(stack, datetime.date(2024, 10, 7), frozenset()) == ("a", "b", "c")
    assert runs_through(stack, datetime.date(2024, 10, 7), frozenset({"b"})) == ("a", "c")


def test_the_canvas_night_is_pinned_by_decision_else_the_newest():
    mosaic = SimpleNamespace(stable_hash="m")
    nights = [datetime.date(2024, 9, 29), datetime.date(2025, 10, 5), datetime.date(2026, 10, 14)]
    assert canvas_night(mosaic, frozendict(), nights) == nights[-1]
    pinned = frozendict({DecisionKey.time_lapse_canvas("m"): DecisionOverride(choice="2025-12-31")})
    assert canvas_night(mosaic, pinned, nights) == nights[1], "the newest night at or before the pin"


def test_the_window_decision_selects_the_nights_to_build():
    mosaic = SimpleNamespace(stable_hash="m")
    nights = [datetime.date(2024, 9, 29), datetime.date(2025, 10, 5), datetime.date(2026, 10, 14)]
    assert nights_window(mosaic, frozendict(), nights) == nights
    window = frozendict({DecisionKey.time_lapse_nights("m"): DecisionOverride(choice="2025-01-01..2025-12-31")})
    assert nights_window(mosaic, window, nights) == [datetime.date(2025, 10, 5)]


def test_stars_fall_back_to_g_then_h_where_broadband_is_missing():
    full, g_only, h_only, none = (0, 0), (0, 1), (1, 0), (1, 1)
    covered = {f: np.zeros((2, 2), bool) for f in "RGBH"}
    for f in "RGBH":
        covered[f][full] = True
    covered["G"][g_only] = covered["H"][g_only] = True
    covered["H"][h_only] = True
    stars = {"R": np.full((2, 2), 1.0, np.float32), "G": np.full((2, 2), 2.0, np.float32),
             "B": np.full((2, 2), 3.0, np.float32), "H": np.full((2, 2), 4.0, np.float32)}
    out = stars_fallback(stars, covered)
    assert out[full].tolist() == [1.0, 2.0, 3.0]
    assert out[g_only].tolist() == [2.0, 2.0, 2.0]
    assert out[h_only].tolist() == [4.0, 4.0, 4.0]
    assert out[none].tolist() == [0.0, 0.0, 0.0]
