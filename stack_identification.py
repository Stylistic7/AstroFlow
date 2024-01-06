import operator

from stack import Stack
import clustering
import file_io


def identify_stacks(spatial_components, plate_solutions, fits_headers):
  print('Identifying stacks...')
  stacks = []
  for name, light_frames in spatial_components:
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
  return stacks