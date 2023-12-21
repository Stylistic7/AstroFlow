import os

from cache import cache


def _add_sample(prev_count, prev_mean, time):
  return (prev_count + 1, (prev_mean * prev_count + time) / (prev_count + 1))


def add_sample(tag, time):
  key = ('throughput_history', tag)
  with cache.transact():
    prev_count, prev_mean = cache.get(key, (0, 0))
    cache[key] = _add_sample(prev_count, prev_mean, time)


def estimate_time(tag, number=1, parallelism=None):
  if not parallelism:
    parallelism = os.cpu_count()

  key = ('throughput_history', tag)
  _, mean = cache.get(key, (0, 0))

  return mean * number / parallelism


if __name__ == '__main__':
  print('foo')