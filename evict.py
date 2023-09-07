import os
import sys

from cache import cache

if __name__ == '__main__':
  # tag = sys.argv[1]
  # print(f'Evicting tag {tag}')
  # cache.evict(tag)

  for k in cache.iterkeys():
    if 'plate_solving' not in k[0]:
      # print(f'would remove key {k}')
      del cache[k]