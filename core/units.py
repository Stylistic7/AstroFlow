import numpy as np


def ra_to_hms(ra):
  hours, remainder = np.divmod(ra * 3600, 15 * 3600)
  minutes, remainder = np.divmod(remainder, 15 * 60)
  seconds = remainder / 15
  return f'{hours:.0f}h {minutes:.0f}m {seconds:.0f}s'


def dec_to_dms(dec):
  degrees, remainder = np.divmod(dec * 3600, 3600)
  minutes, remainder = np.divmod(remainder, 60)
  seconds = remainder
  return f"{degrees:.0f}° {minutes:.0f}' {seconds:.0f}\""
