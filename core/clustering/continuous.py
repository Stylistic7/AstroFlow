import itertools
import operator
from collections import defaultdict
import numpy as np


def cluster_continuous(items, epsilon=0.2, key=operator.itemgetter(1)):
  if len(items) == 0:
    return []

  clusters = []
  items_sorted = sorted(items, key=key)
  curr_point = items_sorted[0]
  curr_cluster = [curr_point]
  for point in items_sorted[1:]:
    if key(point) <= key(curr_point) + epsilon:
      curr_cluster.append(point)
    else:
      clusters.append(curr_cluster)
      curr_cluster = [point]
    curr_point = point
  clusters.append(curr_cluster)

  return clusters


def items_in_distribution(items,
                          items_to_test,
                          max_std_devs=2.0,
                          slop=0.0,
                          key=operator.itemgetter(1)):
  values = [key(i) for i in items]
  mean, std = np.mean(values), np.std(values)
  return [
      i for i in items_to_test
      if abs(key(i) - mean) <= slop or abs(key(i) - mean) / std <= max_std_devs
  ]
