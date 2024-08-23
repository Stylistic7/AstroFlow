import ray

from context import context
import units


def find_missing_tiles(mosaic):
  max_x = max(t[0] for t in mosaic.tile_indices)
  max_y = max(t[1] for t in mosaic.tile_indices)
  print('max_x', max_x, 'max_y', max_y)
  for x in range(max_x + 1):
    for y in range(max_y + 1):
      if (x, y) not in mosaic.tile_indices:
        ra, dec = (mosaic.origin + x * mosaic.basis_vectors[0] +
                   y * mosaic.basis_vectors[1])
        print(f'### {mosaic} is missing tile_indices {(x, y)}, '
              f'RA {units.ra_to_hms(ra)} Dec {units.dec_to_dms(dec)}')


@ray.remote
def make_suggestions():
  for mosaic in context.mosaics.values():
    find_missing_tiles(mosaic)
