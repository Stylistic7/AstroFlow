import os
import sys

from cache import cache

if __name__ == '__main__':
  tag = sys.argv[1]
  print(f'Evicting tag {tag}')
  cache.evict(tag)