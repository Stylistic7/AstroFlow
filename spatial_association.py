import itertools
import io
import os
from operator import itemgetter
from collections import Counter
import numpy as np
from scipy import linalg
from sklearn import mixture
import astropy.io.fits as pyfits
from shapely.geometry import Polygon

import ngc2000_catalog
from distance import distance
import plate_solving

from cache import cache

MIN_COMPONENT_MEMBERS = 5


def iou(a, b):
  return a.intersection(b).area / a.union(b).area


def ra_dec_string(ra, dec):
  def round_to_nearest_tenth(number):
    return int(round(number * 10.0)) / 10.0

  ra = round_to_nearest_tenth(ra)
  dec = round_to_nearest_tenth(dec)

  return f'RA_{ra:.1f}_DEC_{dec:.1f}'


def canonical_target_string(position):
  ra, dec = position

  distance, entry = ngc2000_catalog.find_best_object(ra, dec)
  if entry and entry['name']:
    return entry['name']
  else:
    return ra_dec_string(ra, dec)


# @cache.memoize(tag='spatial_association')
def find_components(filenames, min_component_members=MIN_COMPONENT_MEMBERS):
  footprints = {
      f: Polygon(p[1])
      for f, p in cache['plate_solutions'].items() if f in filenames
  }

  targets = []
  last_printed_percent_done = 0
  for i, (filename, footprint) in enumerate(footprints.items()):
    found = False
    for target in targets:
      if all(iou(footprint, footprints[f]) > 0.7 for f in target):
        target.append(filename)
        found = True
        break

    if not found:
      targets.append([filename])

    percent_done = 100.0 * i / len(filenames)
    percent_done = 10.0 * round(percent_done / 10.0)
    if percent_done > last_printed_percent_done:
      print(f'Finding components is {percent_done:.0f}% done.')
      last_printed_percent_done = percent_done

  targets = [t for t in targets if len(t) > MIN_COMPONENT_MEMBERS]

  def get_name(target):
    position, _, _ = cache['plate_solutions'][target[0]]
    return canonical_target_string(position)

  targets = [(get_name(t), t) for t in targets]

  for name, filenames in targets:
    print(f'Target {name} has {len(filenames)} files')

  target_string_counts = Counter(name for name, _ in targets)

  output_targets = []
  for name, filenames in targets:
    position = cache['plate_solutions'][filenames[0]][0]
    if target_string_counts[name] > 1:
      new_name = f'{name}_{ra_dec_string(*position)}'
      output_targets.append((new_name, filenames))
    else:
      output_targets.append((name, filenames))

  output_targets = [(name, list(sorted(filenames)))
                    for name, filenames in output_targets]

  return list(sorted(output_targets))
