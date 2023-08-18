import itertools
import io
import os
from operator import itemgetter
from collections import Counter
import diskcache
import numpy as np
from scipy import linalg
from sklearn import mixture
import astropy.io.fits as pyfits

import ngc2000_catalog
from distance import distance
import plate_solving

cache = diskcache.Cache(os.path.join(os.getcwd(), '.cache'))

MIN_COMPONENT_MEMBERS = 10


def get_iou(ground_truth, pred):
  # top left to bottom right corner
  left_gt = np.min([p[0] for p in ground_truth])
  top_gt = np.min([p[1] for p in ground_truth])
  right_gt = np.max([p[0] for p in ground_truth])
  bottom_gt = np.max([p[1] for p in ground_truth])

  left_pred = np.min([p[0] for p in pred])
  top_pred = np.min([p[1] for p in pred])
  right_pred = np.max([p[0] for p in pred])
  bottom_pred = np.max([p[1] for p in pred])

  ground_truth = [left_gt, top_gt, right_gt, bottom_gt]
  pred = [left_pred, top_pred, right_pred, bottom_pred]

  # coordinates of the area of intersection.
  ix1 = np.maximum(ground_truth[0], pred[0])
  iy1 = np.maximum(ground_truth[1], pred[1])
  ix2 = np.minimum(ground_truth[2], pred[2])
  iy2 = np.minimum(ground_truth[3], pred[3])

  # Intersection height and width.
  i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
  i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))

  area_of_intersection = i_height * i_width

  # Ground Truth dimensions.
  gt_height = ground_truth[3] - ground_truth[1] + 1
  gt_width = ground_truth[2] - ground_truth[0] + 1

  # Prediction dimensions.
  pd_height = pred[3] - pred[1] + 1
  pd_width = pred[2] - pred[0] + 1

  area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection

  iou = area_of_intersection / area_of_union

  return iou


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


@cache.memoize(tag='spatial_association')
def find_components(filenames, min_component_members=MIN_COMPONENT_MEMBERS):
  footprints = {f: plate_solving.solve_field(f)[1] for f in filenames}

  targets = []
  for filename, footprint in footprints.items():
    found = False
    for target in targets:
      if all(get_iou(footprint, footprints[f]) > 0.7 for f in target):
        target.append(filename)
        found = True
        break

    if not found:
      targets.append([filename])

  targets = [t for t in targets if len(t) > MIN_COMPONENT_MEMBERS]

  def get_name(target):
    position, _, _ = plate_solving.solve_field(target[0])
    return canonical_target_string(position)

  targets = [(get_name(t), t) for t in targets]

  for name, filenames in targets:
    print(f'Target {name} has {len(filenames)} files')

  target_string_counts = Counter(name for name, _ in targets)

  output_targets = []
  for name, filenames in targets:
    position = plate_solving.solve_field(filenames[0])[0]
    if target_string_counts[name] > 1:
      new_name = f'{name}_{ra_dec_string(*position)}'
      output_targets.append((new_name, filenames))
    else:
      output_targets.append((name, filenames))

  output_targets = [(name, list(sorted(filenames)))
                    for name, filenames in output_targets]

  return list(sorted(output_targets))
