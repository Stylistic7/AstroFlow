import itertools
import operator


def cluster_discrete(items, key=operator.itemgetter(1)):
  items = sorted(items, key=key)
  return [list(g) for _, g in itertools.groupby(items, key=key)]
