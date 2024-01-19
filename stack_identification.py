import operator
import ray

from stack import Stack
import clustering
import file_io
import new_dependency_graph as dg


@ray.remote
def identify_stacks_ray(input):
  print('Identifying stacks...')
  groups = input.groups
  fits_headers = ray.get(input.fits_headers)
  plate_solutions = ray.get(input.plate_solutions)

  stacks = []
  for name, light_frames in groups:
    scales = [(f, plate_solutions[f].image_scale) for f in light_frames]
    scale_clusters = clustering.cluster_1d(scales, key=operator.itemgetter(1))
    for scale_cluster in scale_clusters:
      scale_light_frames, _ = zip(*scale_cluster)
      sizes = [(f, fits_headers[f]['NAXIS1'] * fits_headers[f]['NAXIS2'])
               for f in scale_light_frames]
      size_clusters = clustering.cluster_1d(sizes, key=operator.itemgetter(1))
      for size_cluster in size_clusters:
        size_light_frames, _ = zip(*size_cluster)
        for filter, filter_light_frames in file_io.group_by_fits_keyword(
            fits_headers, size_light_frames, 'FILTER'):
          stack = Stack(target=name,
                        filter=filter,
                        light_frames=filter_light_frames)
          stack.store_metadata(fits_headers, plate_solutions)
          stacks.append(stack)

  print(f'Identified {len(stacks)} stacks')
  return dg.ResultDict(ran=True, run_successors=True, result=stacks)