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
