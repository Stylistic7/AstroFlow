from cache import cache
import dependency_graph as dg


def get_sorted_elements(prev_results, task_name, elements, result):
  return sorted(elements[id] for (name, id), status in prev_results.items()
                if name == task_name and id in elements and status == result)


if __name__ == '__main__':
  prev_results = cache['prev_results']
  stacks = cache['stacks']
  targets = cache['targets']

  print('Successful stacks:')
  for stack in get_sorted_elements(prev_results, 'stack_subframes', stacks,
                                   dg.Result.SUCCESS):
    print(stack)

  print('\nFailing stacks:')
  for stack in get_sorted_elements(prev_results, 'stack_subframes', stacks,
                                   dg.Result.FAILURE):
    print(stack)

  print('\nNever-run stacks:')
  for stack in get_sorted_elements(prev_results, 'stack_subframes', stacks,
                                   dg.Result.NEVER_RUN):
    print(stack)

  print('\nSuccessful stack post_processing:')
  for stack in get_sorted_elements(prev_results, 'post_process_stack', stacks,
                                   dg.Result.SUCCESS):
    print(stack)

  print('\nFailing stack post_processing:')
  for stack in get_sorted_elements(prev_results, 'post_process_stack', stacks,
                                   dg.Result.FAILURE):
    print(stack)

  print('\nNever-run stack post_processing:')
  for stack in get_sorted_elements(prev_results, 'post_process_stack', stacks,
                                   dg.Result.NEVER_RUN):
    print(stack)