import itertools

import numpy as np
from scipy import linalg
from sklearn import mixture


def fit_n_components(n_components, positions):
  gmm = mixture.GaussianMixture(n_components=n_components,
                                random_state=0).fit(positions)
  print('GMM means are', gmm.means_)
  print('GMM covariances are', gmm.covariances_)
  print('GMM covariance norms are',
        [np.linalg.norm(c) for c in gmm.covariances_])

  for m, c in zip(gmm.means_, gmm.covariances_):
    print(f'Gaussian with mean {m} has norm {np.linalg.norm(c)}')

  pred = gmm.predict(positions)
  print('GMM predictions are', pred)
  for g in set(pred):
    print(f'Gaussian {g} has {list(pred).count(g)} members')

  return gmm


def check_fit(gmm):
  for c in gmm.covariances_:
    if np.linalg.norm(c) > 1.0:
      return False

  return True


def remove_small_components(gmm, positions):
  pred = gmm.predict(positions)
  print('GMM predictions are', pred)
  for g in set(pred):
    n_members = list(pred).count(g)
    print(f'Gaussian {g} has {n_members} members')
    if n_members < 5:
      pred = [p if p != g else -1 for p in pred]

  return pred


def find_components(positions):
  gmm = None
  for i in range(1, 10):
    gmm = fit_n_components(i, positions)
    if check_fit(gmm):
      print(f'Found acceptable solution with {i} components.')
      return remove_small_components(gmm, positions)
