import os
import sys
import math

from cache import cache

if __name__ == '__main__':
  # tag = sys.argv[1]
  # print(f'Evicting tag {tag}')
  # cache.evict(tag)

  total_size = 0
  for k in cache.iterkeys():
    value = cache[k]
    total_size += len(repr(value))
    len_value_megabytes = len(repr(value)) / 1024.0 / 1024.0
    if len_value_megabytes > 1.0:
      print(f'cache key {k} has size {len_value_megabytes:.03f} MB')

    # if 'plate_solving' not in k[0]:
    #   # print(f'would remove key {k}')
    #   del cache[k]

  total_size_megabytes = total_size / 1024.0 / 1024.0
  print(f'Total size is {total_size_megabytes:.03f} MB')