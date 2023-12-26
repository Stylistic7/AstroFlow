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


# TODO: This function accepts a list of ids, and a dictionary from id to Polygon.
# It could be simplified to accept a list of any objects that have an id and
# a footprint property.
#
# TODO: Add tests.
def find_groups_by_footprint(ids,
                             footprints,
                             min_component_members=MIN_COMPONENT_MEMBERS):
  groups = []
  last_printed_percent_done = 0
  for i, id in enumerate(ids):
    found = False
    footprint = footprints[id]
    for group in groups:
      if iou(footprint, footprints[group[0]]) > 0.6:
        group.append(id)
        found = True
        break

    if not found:
      groups.append([id])

    percent_done = 100.0 * i / len(ids)
    percent_done = 10.0 * round(percent_done / 10.0)
    if percent_done > last_printed_percent_done:
      print(f'Finding groups is {percent_done:.0f}% done.')
      last_printed_percent_done = percent_done

  groups = [t for t in groups if len(t) >= min_component_members]

  return groups


def find_components(light_frames,
                    plate_solutions,
                    fits_headers,
                    min_component_members=MIN_COMPONENT_MEMBERS):
  hashes = sorted(light_frames, key=lambda h: fits_headers[h]['DATE-OBS'])

  # TODO: This is probably expensive, and gets repeated in many places. Maybe
  # it should be done just once, and the footprints stored in Polygon format.
  # Do we ever need the footprint as a list? I suppose we can always just get
  # the list of exterior coords from the Polygon.
  footprints = {h: Polygon(plate_solutions[h].footprint) for h in hashes}

  targets = find_groups_by_footprint(hashes, footprints, min_component_members)

  def get_name(target):
    plate_solution = plate_solutions[target[0]]
    return position_string.canonical_target_string(plate_solution.center)

  def get_complex_name(hash):
    plate_solution = plate_solutions[hash]
    name = position_string.canonical_target_string(plate_solution.center)
    ra_dec = position_string.ra_dec_string(*plate_solution.center)
    square_degrees = Polygon(plate_solution.footprint).area
    return f'{name}_{ra_dec}_{square_degrees:0.2f}_sq_deg'

  targets = [(get_name(t), list(sorted(t))) for t in targets]

  return targets


if __name__ == '__main__':
  find_components(cache['light_frames'])
