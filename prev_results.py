from cache import cache
import dependency_graph as dg


def get_prev_result(task_name, id):
  prev_results = cache.get('prev_results', {})
  return prev_results.get((task_name, id), (dg.Result.NEVER_RUN, None))


def get_prev_result_object(task_name, id):
  result, obj = get_prev_result(task_name, id)
  if result == dg.Result.SUCCESS:
    return obj


# result = (dg.Result.SUCCESS, {
#     ('RGB', 'registered'): {
#         'stack_ids': ['abc123', 'abc456', 'abc789'],
#         'filename': '/foo/bar/RGB.xisf'
#     },
#     ('H', 'starless'): {
#         'stack_ids': ['abc_h'],
#         'filename': '/foo/bar/H.xisf'
#     }
# })

# result = (dg.Result.SUCCESS, {
#     'outputs': [{
#         'filter': 'RGB',
#         'image_type': 'registered',
#         'stack_ids': ['abc123', 'abc456', 'abc789'],
#         'filename': '/foo/bar/RGB.xisf'
#     }, {
#         'filter': 'H',
#         'image_type': 'starless',
#         'stack_ids': ['abc_h'],
#         'filename': '/foo/bar/H.xisf'
#     }]
# })
