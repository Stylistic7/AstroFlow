import itertools
import io
from operator import itemgetter
from collections import Counter

import numpy as np
from scipy import linalg
from sklearn import mixture
import astropy.io.fits as pyfits

import ngc2000_catalog
from distance import distance

MIN_COMPONENT_MEMBERS = 10
LARGEST_ACCEPTABLE_COVARIANCE_AXIS = 0.1  # degrees


def mahalanobis_distance(x, mean, cov):
  M = (x - mean) @ np.linalg.inv(cov) @ np.atleast_2d(x - mean).T
  return np.sqrt(M.item())


def fit_n_components(n_components, positions):
  gmm = mixture.GaussianMixture(n_components=n_components,
                                random_state=0).fit(positions)
  # print('GMM means are', gmm.means_)
  # print('GMM covariances are', gmm.covariances_)
  # print('GMM covariance norms are',
  #       [np.linalg.norm(c) for c in gmm.covariances_])

  for m, c in zip(gmm.means_, gmm.covariances_):
    print(f'Gaussian with mean {m} has norm {np.linalg.norm(c)}')

  pred = gmm.predict(positions)
  # print('GMM predictions are', pred)
  # for g in set(pred):
  #   print(f'Gaussian {g} has {list(pred).count(g)} members')

  return gmm


def check_fit(gmm, positions):
  # TODO: the outlier threshold and the outlier fraction need to be tuned
  num_outliers = 0
  for ra, dec in positions:
    min_distance = np.min([
        distance(ra, dec, mean_ra, mean_dec)
        for mean_ra, mean_dec in gmm.means_
    ])
    # print(f'ra {ra} dec {dec} min distance {min_distance}')
    if min_distance > 0.75:  # degrees
      num_outliers += 1

    # min_mahalanobis_distance = np.min([
    #     mahalanobis_distance(x, mean, cov)
    #     for mean, cov in zip(gmm.means_, gmm.covariances_)
    # ])
    # # print('min_mahalanobis_distance', min_mahalanobis_distance)
    # if min_mahalanobis_distance > 0.5:
    #   num_outliers += 1

  # print(f'Found {num_outliers} outliers / {len(positions)} positions')
  outlier_fraction = float(num_outliers) / len(positions)
  print('outlier_fraction', outlier_fraction)
  return outlier_fraction == 0.0

  # print('Entering check_fit')
  # for c in gmm.covariances_:
  #   print('covariance matrix is', c)
  #   eigenvalues, _ = np.linalg.eig(c)
  #   print('eigenvalues are', eigenvalues)
  #   max_axis_length = np.sqrt(np.max(eigenvalues))
  #   print('max_axis_length', max_axis_length)
  #   if max_axis_length > LARGEST_ACCEPTABLE_COVARIANCE_AXIS:
  #     return False

  # return True


def remove_small_components(gmm, ids, positions, min_component_members):
  pred = gmm.predict(positions)
  # print('GMM predictions are', pred)
  for g in set(pred):
    n_members = list(pred).count(g)
    # print(f'Gaussian {g} has {n_members} members')
    if n_members < min_component_members:
      pred = [p if p != g else -1 for p in pred]

  # TODO: Don't leak the component numbers outside this module.
  return [(i, p) for i, p in zip(ids, pred) if p != -1]


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


def as_groups(gmm, filename_and_pred):
  filename_and_pred = sorted(filename_and_pred, key=itemgetter(1))
  groups = [(gmm.means_[component],
             canonical_target_string(
                 gmm.means_[component]), list(map(itemgetter(0), list(g))))
            for component, g in itertools.groupby(filename_and_pred,
                                                  key=itemgetter(1))]

  target_string_counts = Counter(s for _, s, _ in groups)
  print('target_string_counts', target_string_counts)

  output_groups = []
  for position, name, files in groups:
    if target_string_counts[name] > 1:
      print(f'target name {name} is overloaded')
      new_name = f'{name}_{ra_dec_string(*position)}'
      output_groups.append((new_name, files))
    else:
      output_groups.append((name, files))

  return output_groups


def find_components(ids,
                    positions,
                    min_component_members=MIN_COMPONENT_MEMBERS):
  gmm = None
  for i in range(1, 100000):
    gmm = fit_n_components(i, positions)
    if check_fit(gmm, positions):
      print(f'Found acceptable solution with {i} components.')
      filename_and_pred = remove_small_components(gmm, ids, positions,
                                                  min_component_members)
      return as_groups(gmm, filename_and_pred)
    else:
      print(f'Rejected solution with {i} components')
