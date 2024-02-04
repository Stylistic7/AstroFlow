import itertools
from collections import defaultdict


def cluster_discrete(items, key):
  items = sorted(items, key=key)
  return defaultdict(list, {
      k: list(g) for k, g in itertools.groupby(items, key=key)
  })


def cluster_continuous(items, epsilon=0.2, key=lambda x: x):
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
