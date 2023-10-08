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
import position_string
from distance import distance
import plate_solving

from cache import cache

# TODO: Bring all knobs up here.
MIN_COMPONENT_MEMBERS = 5


def iou(a, b):
  return a.intersection(b).area / a.union(b).area


def find_components(plate_solutions,
                    fits_headers,
                    min_component_members=MIN_COMPONENT_MEMBERS):
  # Filter out images that couldn't be plate-solved.
  plate_solutions = {f: p for f, p in plate_solutions.items() if p is not None}

  filenames = sorted(plate_solutions.keys(),
                     key=lambda x: fits_headers[x]['DATE-OBS'])
  footprints = {f: Polygon(plate_solutions[f].footprint) for f in filenames}

  targets = []
  last_printed_percent_done = 0
  for i, filename in enumerate(filenames):
    found = False
    footprint = footprints[filename]
    for target in targets:
      # if all(iou(footprint, footprints[f]) > 0.6 for f in target):
      if iou(footprint, footprints[target[0]]) > 0.6:
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

  targets = [t for t in targets if len(t) >= min_component_members]

  def get_name(target):
    plate_solution = plate_solutions[target[0]]
    return position_string.canonical_target_string(plate_solution.center)

  targets = [(get_name(t), t) for t in targets]

  target_string_counts = Counter(name for name, _ in targets)

  output_targets = {}
  for name, filenames in targets:
    center = plate_solutions[filenames[0]].center
    if target_string_counts[name] > 1:
      new_name = f'{name}_{position_string.ra_dec_string(*center)}'
      output_targets[new_name] = list(sorted(filenames))
    else:
      output_targets[name] = list(sorted(filenames))

  for name, filenames in output_targets.items():
    print(f'Target {name} has {len(filenames)} files')

  return output_targets


if __name__ == '__main__':
  find_components(cache['light_frames'])
