"""A frame's moment: the middle of its exposure, as naive UTC."""
from datetime import datetime, timedelta, timezone


def utc(value) -> datetime:
    """DATE-OBS as a naive UTC datetime: ISO text with or without a zone (a
    trailing Z included), or a datetime."""
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        dt = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def mid_exposure(header) -> datetime:
    return utc(header["DATE-OBS"]) + timedelta(seconds=float(header.get("EXPTIME") or 0) / 2)
