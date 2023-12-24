import os
import itertools


def round_robin_by_directory(filenames, cluster=3):
  filenames = sorted(filenames)
  dirs = sorted(set(os.path.dirname(f) for f in filenames))

  while filenames:
    for dir in dirs:
      files_in_dir = [f for f in filenames if os.path.dirname(f) == dir]
      for filename in files_in_dir[:cluster]:
        yield filename
        filenames.remove(filename)


def chunked(iterable, size=10):
  iterator = iter(iterable)
  for first in iterator:
    yield itertools.chain([first], itertools.islice(iterator, size - 1))