import os
import sys
import math

from context import get_cache

cache = get_cache()

if __name__ == '__main__':

  def starts_with(big_tuple, small_tuple):
    if not small_tuple:
      return True
    if len(small_tuple) > len(big_tuple):
      return False
    return big_tuple[:len(small_tuple)] == small_tuple

  # delete_list = [('mosaic_thumbnail_image',), ('merged_images',),
  #                ('placed_image',), ('mosaic_viewbox',)]
  # for k in cache.iterkeys():
  #   if any(starts_with(k, d) for d in delete_list):
  #     print('deleting key', k)
  #     del cache[k]

  keep_list = [('image_metadata',)]
  for k in cache.iterkeys():
    if not any(starts_with(k, d) for d in keep_list):
      print('deleting key', k)
      del cache[k]
