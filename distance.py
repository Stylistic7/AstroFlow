import numpy as np
from numpy import fabs, sin, cos, arctan2, sqrt


# TODO: Name this better.
def distance(ra_1, dec_1, ra_2, dec_2, degrees=True):
  if degrees:
    ra_1, dec_1 = np.radians(ra_1), np.radians(dec_1)
    ra_2, dec_2 = np.radians(ra_2), np.radians(dec_2)

  # lambda is ra, sigma is dec
  ra_diff = fabs(ra_1 - ra_2)
  dec_diff = fabs(dec_1 - dec_2)

  t1 = cos(dec_2) * sin(ra_diff)
  t2 = cos(dec_1) * sin(dec_2) - sin(dec_1) * cos(dec_2) * cos(ra_diff)
  numerator = sqrt(t1**2.0 + t2**2.0)

  denominator = sin(dec_1) * sin(dec_2) + cos(dec_1) * cos(dec_2) * cos(ra_diff)

  result = arctan2(numerator, denominator)

  if degrees:
    return np.degrees(result)
  else:
    return result