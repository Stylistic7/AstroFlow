from ray_infra import ray_local

import ray
import numpy as np


@ray.remote
def f(x):
  return x * x


@ray.remote
def func(large_arg, i):
  print(f'type(large_arg) {type(large_arg)}')
  return len(large_arg) + i


futures = [f.remote(i) for i in range(4)]
print(ray.get(futures))  # [0, 1, 4, 9]

large_arg = np.zeros(1024 * 1024)

# 1 copy of large_arg is stored in the object store.
large_arg_ref = ray.put(large_arg)
outputs = ray.get([func.remote(large_arg_ref, i) for i in range(10)])
