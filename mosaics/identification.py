import traceback
import structlog
from spherical_geometry.polygon import SphericalPolygon, SingleSphericalPolygon
from spherical_geometry.vector import lonlat_to_vector
import ray
import numpy as np
import random

from context import context
from mosaic import Mosaic
import position_string
from geometry import footprints
from hashing import hash_list
import units

from . import ransac

logger = structlog.get_logger()
log = logger


def _identify_overlapping_targets(targets):
  if not targets:
    return []

  def roughly_same_area(a, b):
    ratio = a.area() / b.area()
    return ratio > 0.9 and ratio < 1.1

  # TODO: Targets should be really be checked that they overlap mostly along
  # one axis (e.g. the long edge or the short edge), within some amount (say
  # 5-30%).

  # Sort the targets by their center RA.
  target_ids = list(sorted(targets.keys(), key=lambda t: targets[t].center[0]))
  log.info('Identifying overlapping targets', num_targets=len(targets))

  mosaics = []
  for target_id in target_ids:
    matching_mosaic_indices = []
    for i, mosaic in enumerate(mosaics):
      compatible_area = roughly_same_area(targets[target_id].footprint,
                                          targets[mosaic[0]].footprint)
      compatible_overlap = any(
          targets[target_id].footprint.intersects_poly(targets[t].footprint)
          for t in mosaic)
      if compatible_area and compatible_overlap:
        matching_mosaic_indices.append(i)

    if not matching_mosaic_indices:
      print(f'### Creating new mosaic to contain target_id {target_id}')
      mosaics.append([target_id])
    elif len(matching_mosaic_indices) == 1:
      print(f'### Adding target_id {target_id} to existing mosaic')
      mosaics[matching_mosaic_indices[0]].append(target_id)
    elif len(matching_mosaic_indices) > 1:
      print(f'### Target {targets[target_id]} fits into '
            f'{len(matching_mosaic_indices)} mosaics')
      print(f'### Before merging mosaics, there are {len(mosaics)} mosaics')
      merged_mosaic = [target_id]
      for i in sorted(matching_mosaic_indices, reverse=True):
        merged_mosaic.extend(mosaics[i])
        del mosaics[i]
      print(f'### Merged mosaic contains {len(merged_mosaic)} targets')
      mosaics.append(merged_mosaic)
      print(f'### After merging mosaics, there are {len(mosaics)} mosaics')

  num_grouped_targets = sum(len(m) for m in mosaics)
  num_ungrouped_targets = len(targets) - num_grouped_targets
  log.info('Done grouping overlapping targets',
           num_mosaics=len(mosaics),
           num_grouped_targets=num_grouped_targets,
           num_ungrouped_targets=num_ungrouped_targets)

  mosaics = [m for m in mosaics if len(m) > 1]

  return mosaics


@ray.remote
def identify_mosiacs():
  targets = context.targets
  print(f'### Identifying mosaics from {len(targets)} targets')

  mosaics = _identify_overlapping_targets(targets)
  if not mosaics:
    print('### No overlapping targets found')
    context.mosaics = {}
    return

  print(f'Found {len(mosaics)} mosaics')

  output_mosaics = {}
  for mosaic in mosaics:
    target_ids, tile_indices, origin, basis_vectors = ransac.ransac_2d(
        mosaic, targets, plot=False, plot_title='foo')

    if not target_ids:
      print('ransac_2d failed')
      target_ids, tile_indices, origin, basis_vectors = ransac.ransac_1d(
          mosaic, targets)

      if not target_ids:
        print('ransac_1d also failed')
        continue

    print(f'Mosaic composed of {len(target_ids)} targets:')
    for target_id in target_ids:
      print(f'  {targets[target_id].name} id {target_id}')

    try:
      # TODO: Validate all targets here.

      stable_hash = targets[sorted(
          target_ids, key=lambda t: targets[t].oldest_date)[0]].stable_hash
      content_hash = hash_list(targets[t].content_hash for t in target_ids)

      oldest_date = min(targets[t].oldest_date for t in target_ids),
      newest_date = min(targets[t].newest_date for t in target_ids),

      centers = [targets[t].center for t in target_ids]
      mean_center = np.mean(centers, axis=0)
      center = list(mean_center)
      name = position_string.canonical_target_string(center)

      # This doesn't seem work for polygons with holes in them.
      # TODO: Consider using multi_union but falling back to convex hull when
      # it doesn't work?
      # old_footprint = next(
      #     SphericalPolygon.multi_union([
      #         SphericalPolygon(targets[target_id].footprint)
      #         for target_id in target_ids
      #     ]).points).tolist()

      all_points = np.concatenate(
          [targets[target_id].footprint.points for target_id in target_ids])

      # TODO: There's no way to specify an `inside` point here?
      footprint = SingleSphericalPolygon.convex_hull(all_points)
      footprint = SingleSphericalPolygon.convex_hull(footprint.points)

      # TODO: This is turbo weird, but appears to work. Just randomize the
      # points until the `convex_hull()` operation is successful.
      for i in range(10):
        if SphericalPolygon.self_intersect(footprint.points):
          points = footprint.points
          random.shuffle(points)
          footprint = SingleSphericalPolygon.convex_hull(points)
        else:
          break

      assert not SphericalPolygon.self_intersect(footprint.points)
      projected_footprint = footprints.all_projections(footprint)

      output_mosaic = Mosaic(name=name,
                             stable_hash=stable_hash,
                             content_hash=content_hash,
                             target_ids=target_ids,
                             tile_indices=tile_indices,
                             origin=origin,
                             basis_vectors=basis_vectors,
                             oldest_date=oldest_date,
                             newest_date=newest_date,
                             center=center,
                             footprint=footprint,
                             projected_footprint=projected_footprint)

      print(f'Mosaic has name {name} stable_hash {stable_hash} and '
            f'basis_vectors {basis_vectors}')
      output_mosaics[stable_hash] = output_mosaic
    except Exception as e:
      traceback.print_exc()

  # TODO: Computing the result and storing the result are two different
  # responsibilities. Perhaps separate patterns like this?
  context.mosaics = output_mosaics
