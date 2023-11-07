from cache import cache
import dependency_graph as dg
from astro_flow_direct import PlateSolutions


def get_sorted_elements(prev_results, task_name, elements, result):
  return sorted(elements[id]
                for (name, id), (_, (status, _)) in prev_results.items()
                if name == task_name and id in elements and status == result)


if __name__ == '__main__':
  plate_solutions = cache.get('plate_solutions', {})
  num_exceptions = sum(
      [1 for x in plate_solutions.values() if type(x) is Exception])
  num_missing_files = sum([1 for x in plate_solutions.values() if x is None])
  print(f'There are {len(plate_solutions)} plate solutions')
  print(f'  {num_exceptions} did not solve, and ended with an exception')
  print(f'  {num_missing_files} did not solve, and ended with missing files')

  prev_results = cache.get('prev_results', {})
  stacks = cache.get('stacks', {})
  targets = cache.get('targets', {})

  # TODO: Clean all of this up.
  print('\nSuccessful stacks:')
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

  print('\nSuccessful target post_processing:')
  for target in get_sorted_elements(prev_results, 'post_process_target',
                                    targets, dg.Result.SUCCESS):
    print(target)

  print('\nFailing target post_processing:')
  for target in get_sorted_elements(prev_results, 'post_process_target',
                                    targets, dg.Result.FAILURE):
    print(target)

  print('\nNever-run target post_processing:')
  for target in get_sorted_elements(prev_results, 'post_process_target',
                                    targets, dg.Result.NEVER_RUN):
    print(target)

  for k, mosaic in cache.get('mosaics', {}).items():
    print(f'### mosaic {mosaic.name} {mosaic}')
