import os
import sys
import math

from cache import cache

if __name__ == '__main__':
  # # tag = sys.argv[1]
  # # print(f'Evicting tag {tag}')
  # # cache.evict(tag)

  # # total_size = 0
  # for k in cache.iterkeys():
  #   # value = cache[k]
  #   # total_size += len(repr(value))
  #   # len_value_megabytes = len(repr(value)) / 1024.0 / 1024.0
  #   # if len_value_megabytes > 1.0:
  #   #   print(f'cache key {k} has size {len_value_megabytes:.03f} MB')

  #   # if 'plate_solving' not in k[0]:
  #   #   # print(f'would remove key {k}')
  #   #   del cache[k]

  #   if k != 'plate_solutions':
  #     del cache[k]

  # # total_size_megabytes = total_size / 1024.0 / 1024.0
  # # print(f'Total size is {total_size_megabytes:.03f} MB')

  # plate_solutions = cache['plate_solutions']
  # image_hashes = cache['image_hashes']

  # cache.clear()

  # # TODO: Implement quarantining
  # # plate_solutions = {f: p for f, p in plate_solutions.items() if p is not None}

  # cache['plate_solutions'] = plate_solutions
  # cache['image_hashes'] = image_hashes

  # allow_list = [
  #     'plate_solutions', 'image_hashes', 'thumbnail_image', 'thumbnail_video',
  #     'stack_thumbnail_image', 'stack_thumbnail_video', 'trial_flat_correction'
  # ]

  allow_list = ['image_hashes', 'plate_solutions', 'fits_headers']

  for k in cache.iterkeys():
    # print('looking at key', k)
    if any(a in k for a in allow_list):
      continue

    try:
      print('deleting key', k)
      del cache[k]
    except:
      pass
