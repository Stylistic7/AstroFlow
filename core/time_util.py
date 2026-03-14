import ephem  # type: ignore
import datetime
from dateutil.parser import parse
import pytz  # type: ignore
from beartype import beartype

kSanFranciscoLat = 37.7749
kSanFranciscoLon = -122.4194


@beartype
def parse_local_datetime(s: str) -> datetime.datetime:
    local = pytz.timezone("America/Los_Angeles")
    naive = parse(s)
    local_dt = local.localize(naive, is_dst=None)
    utc_dt = local_dt.astimezone(pytz.utc)
    return utc_dt


@beartype
def parse_utc_datetime(s: str) -> datetime.datetime:
    return pytz.utc.localize(parse(s))


@beartype
def period_overlap(
    p1: tuple[datetime.datetime, datetime.datetime],
    p2: tuple[datetime.datetime, datetime.datetime],
) -> float:
    latest_start = max(p1[0], p2[0])
    earliest_end = min(p1[1], p2[1])
    delta = (earliest_end - latest_start).total_seconds()
    return max(0, delta)


def get_next_twilight_time() -> datetime.date:
    observer = ephem.Observer()
    observer.date = datetime.datetime.now()
    observer.lat = str(kSanFranciscoLat)  # Note that lat should be in string format
    observer.lon = str(kSanFranciscoLon)  # Note that lon should be in string format
    observer.elev = 20

    # To get U.S. Naval Astronomical Almanac values, use these settings
    observer.pressure = 0
    observer.horizon = "-0:34"

    sunrise = observer.previous_rising(ephem.Sun())  # Sunrise
    noon = observer.next_transit(ephem.Sun(), start=sunrise)  # Solar noon
    sunset = observer.next_setting(ephem.Sun())  # Sunset

    # We relocate the horizon to get twilight times
    observer.horizon = "-18"  # - 6=civil twilight, -12=nautical, -18=astronomical
    twilight = observer.next_rising(ephem.Sun(), use_center=True)

    return pytz.utc.localize(twilight.datetime())


@beartype
def before_twilight(margin_minutes: float = 30) -> bool:
    twilight_dt = get_next_twilight_time()
    print("Next astronomical twilight starts at", twilight_dt, "UTC")
    twilight_dt += datetime.timedelta(minutes=margin_minutes)

    return datetime.datetime.now(pytz.utc) < twilight_dt


if __name__ == "__main__":
    print("next twilight time", get_next_twilight_time())
