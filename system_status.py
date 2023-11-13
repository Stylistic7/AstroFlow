import sys
from cache import cache
import objsize

import dependency_graph as dg
from astro_flow_direct import PlateSolutions

MEGABYTE = 1024.0 * 1024.0


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
  mosaics = cache.get('mosaics', {})

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

  print('\nSuccessful mosaic post_processing:')
  for mosaic in get_sorted_elements(prev_results, 'post_process_mosaic',
                                    mosaics, dg.Result.SUCCESS):
    print(mosaic)

  print('\nFailing mosaic post_processing:')
  for mosaic in get_sorted_elements(prev_results, 'post_process_mosaic',
                                    mosaics, dg.Result.FAILURE):
    print(mosaic)

  print('\nNever-run mosaic post_processing:')
  for mosaic in get_sorted_elements(prev_results, 'post_process_mosaic',
                                    mosaics, dg.Result.NEVER_RUN):
    print(mosaic)

  thumbnail_image_size = sum(
      objsize.get_deep_size(cache[k]) for k in cache
      if k[0] == 'thumbnail_image')
  print(f'\nSize of thumbnail images '
        f'{thumbnail_image_size / MEGABYTE:.3f} MB')

  stack_thumbnail_image_size = sum(
      objsize.get_deep_size(cache[k]) for k in cache
      if k[0] == 'stack_thumbnail_image')
  print(f'\nSize of stack_thumbnail images '
        f'{stack_thumbnail_image_size / MEGABYTE:.3f} MB')

  stack_thumbnail_video_size = sum(
      objsize.get_deep_size(cache[k]) for k in cache
      if k[0] == 'stack_thumbnail_video')
  print(f'\nSize of stack_thumbnail videos '
        f'{stack_thumbnail_video_size / MEGABYTE:.3f} MB')

  plate_solutions_size = objsize.get_deep_size(
      cache['plate_solutions']) / MEGABYTE
  print(f'\nSize of plate solutions {plate_solutions_size:.3f} MB')

  fits_headers_size = objsize.get_deep_size(cache['fits_headers']) / MEGABYTE
  print(f'\nSize of fits headers {fits_headers_size:.3f} MB')

  # for k, mosaic in cache.get('mosaics', {}).items():
  #   print(f'### mosaic {mosaic.name} {mosaic}')
