import ray
import numpy as np

from context import context


def estimate_time(tag: str, number: int = 1) -> float | None:
  nodes = ray.nodes()
  times = []
  for node in nodes:
    name = node['NodeName']
    num_cpus = node['Resources']['CPU']
    key = ('performance_history', name, tag)
    if history := context.get(key, None):
      print(f'### history {history}')

      # If the number of remaining tasks is less than the number of CPUs,
      # then all of them can run at the same time.
      if number < num_cpus:
        times.append(np.mean(history))
      else:
        times.append(num_cpus / np.mean(history))

  if times:
    return number / np.sum(times)
