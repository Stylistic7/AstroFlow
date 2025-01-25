import os

from context import get_cache

cache = get_cache()

if __name__ == '__main__':

  def starts_with(big_tuple, small_tuple):
    # Check if the small_tuple is empty
    if not small_tuple:
      return True
    # Check if the small_tuple is longer than the big_tuple
    if len(small_tuple) > len(big_tuple):
      return False
    # Check if the beginning segment of big_tuple matches small_tuple
    return big_tuple[:len(small_tuple)] == small_tuple

  num_good = 0
  num_bad = 0
  for k in cache.iterkeys():
    if starts_with(k, ('image_metadata',)):
      metadata = cache[k]
      if os.path.exists(metadata.filename):
        num_good += 1
      else:
        print(f'key {k} has non-existent filename {metadata.filename}')
        del cache[k]
        num_bad += 1

  print(f'### num_good {num_good} num_bad {num_bad}')
