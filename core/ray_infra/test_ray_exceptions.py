import ray


@ray.remote
def remote_fn(i):
    if i == 10:
        raise ValueError("invalid i == 10")
    else:
        return i


# def test_one_exception_parallel_tasks(init_ray):
#   waitables = [remote_fn.remote(i) for i in range(20)]
#   num_tasks = len(waitables)

#   results = []
#   while len(waitables):
#     done, waitables = ray.wait(waitables)
#     try:
#       results.extend(ray.get(done))
#     except Exception as e:
#       print(f'Caught exception {e}')

#   assert len(results) == num_tasks - 1

#   expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19]
#   assert sorted(results) == expected
