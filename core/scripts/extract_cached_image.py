import os
import sys
import math

from context import get_cache

cache = get_cache()

if __name__ == '__main__':
  mosaic_id = 'b1c2c11588a866ac'
  filter = 'H'
  stack_id = '2c71f4db63b27773'
  key = ('placed_image', mosaic_id, filter, stack_id)

  filename = f'/Volumes/Astro/tmp/{'_'.join(key)}.fits'
  open(filename, 'wb').write(cache[key])
  print(f'### Wrote filename {filename}')
