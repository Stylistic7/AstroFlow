import itertools
import io
from operator import itemgetter

import numpy as np
from scipy import linalg
from sklearn import mixture
import astropy.io.fits as pyfits


def fit_n_components(n_components, positions):
  gmm = mixture.GaussianMixture(n_components=n_components,
                                random_state=0).fit(positions)
  # print('GMM means are', gmm.means_)
  # print('GMM covariances are', gmm.covariances_)
  # print('GMM covariance norms are',
  #       [np.linalg.norm(c) for c in gmm.covariances_])

  # for m, c in zip(gmm.means_, gmm.covariances_):
  #   print(f'Gaussian with mean {m} has norm {np.linalg.norm(c)}')

  pred = gmm.predict(positions)
  # print('GMM predictions are', pred)
  # for g in set(pred):
  #   print(f'Gaussian {g} has {list(pred).count(g)} members')

  return gmm


def check_fit(gmm):
  for c in gmm.covariances_:
    if np.linalg.norm(c) > 1.0:
      return False

  return True


def remove_small_components(gmm, ids, positions):
  pred = gmm.predict(positions)
  # print('GMM predictions are', pred)
  for g in set(pred):
    n_members = list(pred).count(g)
    # print(f'Gaussian {g} has {n_members} members')
    if n_members < 5:
      pred = [p if p != g else -1 for p in pred]

  # TODO: Don't leak the component numbers outside this module.
  return [(i, p) for i, p in zip(ids, pred) if p != -1]


def as_groups(filename_and_pred):
  filename_and_pred = sorted(filename_and_pred, key=itemgetter(1))
  return [
      list(map(itemgetter(0), list(g)))
      for _, g in itertools.groupby(filename_and_pred, key=itemgetter(1))
  ]


def find_components(ids, positions):
  gmm = None
  for i in range(1, 10):
    gmm = fit_n_components(i, positions)
    if check_fit(gmm):
      print(f'Found acceptable solution with {i} components.')
      return as_groups(remove_small_components(gmm, ids, positions))


def get_position(wcs_string):
  with pyfits.open(io.BytesIO(wcs_string)) as fits:
    return fits[0].header['CRVAL1'], fits[0].header['CRVAL2']