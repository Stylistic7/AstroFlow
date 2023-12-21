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


def find_components(light_frames,
                    plate_solutions,
                    fits_headers,
                    min_component_members=MIN_COMPONENT_MEMBERS):
  hashes = sorted(light_frames, key=lambda h: fits_headers[h]['DATE-OBS'])
  print('sorted hashes', hashes[0:10])

  # TODO: This is probably expensive, and gets repeated in many places. Maybe
  # it should be done just once, and the footprints stored in Polygon format.
  # Do we ever need the footprint as a list? I suppose we can always just get
  # the list of exterior coords from the Polygon.
  footprints = {h: Polygon(plate_solutions[h].footprint) for h in hashes}

  targets = []
  last_printed_percent_done = 0
  for i, hash in enumerate(hashes):
    found = False
    footprint = footprints[hash]
    for target in targets:
      # if all(iou(footprint, footprints[f]) > 0.6 for f in target):
      if iou(footprint, footprints[target[0]]) > 0.6:
        target.append(hash)
        found = True
        break

    if not found:
      targets.append([hash])

    percent_done = 100.0 * i / len(hashes)
    percent_done = 10.0 * round(percent_done / 10.0)
    if percent_done > last_printed_percent_done:
      print(f'Finding components is {percent_done:.0f}% done.')
      last_printed_percent_done = percent_done

  targets = [t for t in targets if len(t) >= min_component_members]

  def get_name(target):
    plate_solution = plate_solutions[target[0]]
    return position_string.canonical_target_string(plate_solution.center)

  targets = [(get_name(t), t) for t in targets]
  print('### Before renaming')
  for name, hashes in targets:
    print(f'### Target {name} has {len(hashes)} files')

  target_string_counts = Counter(name for name, _ in targets)

  output_targets = {}
  for name, hashes in targets:
    center = plate_solutions[hashes[0]].center
    if target_string_counts[name] > 1:
      new_name = f'{name}_{position_string.ra_dec_string(*center)}'
      # TODO: This can still collide in some cases, resulting in lost stacks.
      print(f'### Target {name} will be renamed {new_name}')
      assert new_name not in output_targets
      output_targets[new_name] = list(sorted(hashes))
    else:
      output_targets[name] = list(sorted(hashes))

  assert len(output_targets) == len(targets)

  for name, hashes in output_targets.items():
    print(f'Target {name} has {len(hashes)} files')

  return output_targets


if __name__ == '__main__':
  find_components(cache['light_frames'])
