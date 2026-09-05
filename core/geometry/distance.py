from numpy import fabs, sin, cos, arctan2, sqrt
from beartype import beartype

from units import Degrees, Radians


# TODO: Name this better.
@beartype
def distance(ra_1: Degrees, dec_1: Degrees, ra_2: Degrees, dec_2: Degrees) -> Degrees:
    """Great-circle distance between two sky positions (Vincenty formula)."""
    ra_1, dec_1 = ra_1.to_radians(), dec_1.to_radians()
    ra_2, dec_2 = ra_2.to_radians(), dec_2.to_radians()

    # lambda is ra, sigma is dec
    ra_diff = fabs(ra_1 - ra_2)

    t1 = cos(dec_2) * sin(ra_diff)
    t2 = cos(dec_1) * sin(dec_2) - sin(dec_1) * cos(dec_2) * cos(ra_diff)
    numerator = sqrt(t1**2.0 + t2**2.0)

    denominator = sin(dec_1) * sin(dec_2) + cos(dec_1) * cos(dec_2) * cos(ra_diff)

    return Radians(arctan2(numerator, denominator)).to_degrees()
