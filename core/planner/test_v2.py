"""v2's pure pieces: tonight's share of a need, and which night a moment
belongs to. The month solve itself is covered by test_allocate and
test_horizon; the end-to-end plan runs in shadow mode against the real cart."""
from datetime import datetime, timezone

from planner.v2 import night_key, quota_tiers


def test_quota_boosts_the_first_hours_and_keeps_the_rest():
    assert quota_tiers([[1.0, 2.0], [3.0, 0.5]], 2.0, boost=3.0) == [[1.0, 6.0], [1.0, 1.5], [2.0, 0.5]]
    assert quota_tiers([[1.0, 2.0], [3.0, 0.5]], 0.25, boost=3.0) == [[0.25, 6.0], [0.75, 2.0], [3.0, 0.5]]
    assert quota_tiers([[1.0, 2.0]], 5.0, boost=3.0) == [[1.0, 6.0]]        # never more than the need
    assert quota_tiers([[1.0, 2.0], [3.0, 0.5]], 0.0) == [[1.0, 2.0], [3.0, 0.5]]   # no quota: v1 as is


def test_night_key_is_the_evening_date_until_local_noon():
    lon = -122.42                                               # UTC-8.2 h solar
    evening = datetime(2026, 10, 13, 4, tzinfo=timezone.utc)    # 20:00 local on Oct 12
    dawn = datetime(2026, 10, 13, 13, tzinfo=timezone.utc)      # 05:00 local on Oct 13
    assert night_key(evening, lon).isoformat() == "2026-10-12"
    assert night_key(dawn, lon).isoformat() == "2026-10-12"
    assert night_key(datetime(2026, 10, 13, 21, tzinfo=timezone.utc), lon).isoformat() == "2026-10-13"
