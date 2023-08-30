import os
import diskcache
import sys

cache = diskcache.Cache(os.path.join(os.getcwd(), '.cache'))

if __name__ == '__main__':
  tag = sys.argv[1]
  print(f'Evicting tag {tag}')
  cache.evict(tag)