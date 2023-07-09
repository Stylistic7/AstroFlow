import datetime
import re
from pathlib import Path
from dateutil.parser import parse


def find_most_recent_color_channels(masters):
  print(f"find_most_recent_color_channels with masters {masters}")
  output = []
  for filt in ['H', 'O', 'S', 'L', 'G', 'R', 'B']:
    matches = list(filter(lambda f: f['filter'] == filt, masters))
    if matches:
      max_date = max(m['date'] for m in matches)
      # FIXME what if there are two rigs for a single color channel?
      most_recent_master = [m for m in matches if m['date'] == max_date]
      output.append(most_recent_master[0])

  return output


def find_reference(masters):
  print(f"find_reference with masters {masters}")
  for rig in ['C11_F7', 'C11', 'RC71', 'TestRig']:
    for filt in ['H', 'O', 'S', 'L', 'G', 'R', 'B']:
      matches = list(
          filter(lambda f: f['rig'] == rig and f['filter'] == filt, masters))
      if len(matches):
        return matches[0]
