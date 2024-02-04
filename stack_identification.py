import operator
import ray

from stack import Stack
import clustering
import file_io


@ray.remote
def identify_stacks_ray(context):
  print('Identifying stacks...')

  fits_headers = ray.get(context.fits_headers)
  plate_solutions = ray.get(context.plate_solutions)
  spatial_groups = ray.get(context.spatial_groups)

  # TODO: This also needs to cluster rotation angle.
  stacks = []
  for name, light_frames in spatial_groups:
    scales = [(f, plate_solutions[f].image_scale) for f in light_frames]
    scale_clusters = clustering.cluster_continuous(scales,
                                                   key=operator.itemgetter(1))
    for scale_cluster in scale_clusters:
      scale_light_frames, _ = zip(*scale_cluster)
      sizes = [(f, fits_headers[f]['NAXIS1'] * fits_headers[f]['NAXIS2'])
               for f in scale_light_frames]
      size_clusters = clustering.cluster_continuous(sizes,
                                                    key=operator.itemgetter(1))
      for size_cluster in size_clusters:
        size_light_frames, _ = zip(*size_cluster)
        for filter, filter_light_frames in file_io.group_by_fits_keyword(
            fits_headers, size_light_frames, 'FILTER'):
          stack = Stack(target=name,
                        filter=filter,
                        light_frames=filter_light_frames)
          stacks.append(stack)

  print(f'Identified {len(stacks)} stacks')
  context.stacks = ray.put(stacks)
  return context
