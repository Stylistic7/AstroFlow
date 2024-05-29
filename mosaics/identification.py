import traceback
import structlog
from spherical_geometry.polygon import SphericalPolygon
import ray
import numpy as np

from new_context import new_context
from mosaic import Mosaic
import position_string
import footprints

from . import ransac

logger = structlog.get_logger()
log = logger


def _identify_overlapping_targets(targets):
  if not targets:
    return []

  def overlap(a, b):
    return a.intersection(b).area() > 0

  def roughly_same_area(a, b):
    ratio = a.area() / b.area()
    return ratio > 0.9 and ratio < 1.1

  # TODO: Targets should be really be checked that they overlap mostly along
  # one axis (e.g. the long edge or the short edge), within some amount (say
  # 5-30%).

  # Sort the targets by their center RA.
  targets = list(sorted(targets, key=lambda t: t.center[0]))
  log.info('Identifying overlapping targets', num_targets=len(targets))

  mosaics = []
  for target in targets:
    found = False
    for mosaic in mosaics:
      all_roughly_same_area = all(
          roughly_same_area(target.footprint_polygon(), m.footprint_polygon())
          for m in mosaic)
      any_overlap = any(
          overlap(target.footprint_polygon(), m.footprint_polygon())
          for m in mosaic)
      if all_roughly_same_area and any_overlap:
        mosaic.append(target)
        found = True
        break

    if not found:
      mosaics.append([target])

  num_grouped_targets = sum(len(m) for m in mosaics)
  num_ungrouped_targets = len(targets) - num_grouped_targets
  log.info('Done grouping overlapping targets',
           num_grouped_targets=num_grouped_targets,
           num_ungrouped_targets=num_ungrouped_targets)

  mosaics = [m for m in mosaics if len(m) > 1]

  return mosaics


@ray.remote
def identify_mosiacs():
  targets = new_context.targets
  print(f'### Identifying mosaics from {len(targets)} targets')

  mosaics = _identify_overlapping_targets(targets.values())
  if not mosaics:
    print('### No overlapping targets found')
    new_context.mosaics = {}
    return

  print(f'Found {len(mosaics)} mosaics')

  output_mosaics = {}
  for mosaic in mosaics:
    target_ids, tile_indices = ransac.ransac(mosaic,
                                             plot=False,
                                             plot_title='foo')

    if not target_ids:
      print('RANSAC failed')
      continue

    print(f'Mosaic composed of {len(target_ids)} targets:')
    for target_id in target_ids:
      print(f'  {targets[target_id].name} id {target_id}')

    try:
      output_mosaic = Mosaic()

      centers = [targets[id].center for id in target_ids]
      mean_center = np.mean(centers, axis=0)
      output_mosaic.center = list(mean_center)

      output_mosaic.name = position_string.canonical_target_string(
          output_mosaic.center)
      print('### target_ids', target_ids)
      output_mosaic.target_ids = target_ids
      output_mosaic.tile_indices = tile_indices
      output_mosaic.footprint = next(
          SphericalPolygon.multi_union([
              SphericalPolygon(targets[target_id].footprint)
              for target_id in target_ids
          ]).points).tolist()
      output_mosaic.projected_footprint = footprints.all_projections(
          SphericalPolygon(output_mosaic.footprint))

      # TODO: Mosaic is mutated here, which is awkward. Mosaic objects should
      # be immutable.
      output_mosaic.id = output_mosaic.hash()

      print(f'Mosaic has name {output_mosaic.name} and id {output_mosaic.id}')
      output_mosaics[output_mosaic.id] = output_mosaic
    except Exception as e:
      traceback.print_exc()

  # TODO: Computing the result and storing the result are two different
  # responsibilities. Perhaps separate patterns like this?
  new_context.mosaics = output_mosaics
