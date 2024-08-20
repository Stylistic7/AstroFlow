from ray_infra import ray_local

import unittest
import ray


@ray.remote
def remote_fn(i):
  if i == 10:
    raise ValueError("invalid i == 10")
  else:
    return i


class TestRayExceptions(unittest.TestCase):

  def test_one_exception_parallel_tasks(self):
    waitables = [remote_fn.remote(i) for i in range(20)]
    num_tasks = len(waitables)

    results = []
    while len(waitables):
      done, waitables = ray.wait(waitables)
      try:
        results.extend(ray.get(done))
      except Exception as e:
        print(f'Caught exception {e}')

    self.assertEqual(len(results), num_tasks - 1)

    expected = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19
    ]
    self.assertEqual(sorted(results), expected)


if __name__ == "__main__":
  unittest.main()
