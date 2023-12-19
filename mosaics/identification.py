import traceback

from shapely import Polygon
import structlog

from mosaic import Mosaic
import position_string

from . import ransac

logger = structlog.get_logger()
log = logger


def identify_overlapping_targets(targets):
  if not targets:
    return []

  def overlap(a, b):
    return a.intersection(b).area > 0

  def roughly_same_area(a, b):
    ratio = a.area / b.area
    return ratio > 0.9 and ratio < 1.1

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

  return mosaics


def identify_mosiacs(targets, plot=False):
  mosaics = identify_overlapping_targets(targets.values())
  if not mosaics:
    return []

  print(f'Found {len(mosaics)} mosaics')

  output_mosaics = []
  for mosaic in mosaics:
    target_ids, tile_indices = ransac.ransac(mosaic,
                                             plot=plot,
                                             plot_title='foo')

    if not target_ids:
      print('RANSAC failed')
      continue

    print(f'Mosaic composed of {len(target_ids)} targets:')
    for target_id in target_ids:
      print(f'  {targets[target_id].name}')

    try:
      output_mosaic = Mosaic()
      output_mosaic.target_ids = target_ids
      output_mosaic.tile_indices = tile_indices
      output_mosaic.store_metadata(targets)
      output_mosaic.name = position_string.canonical_target_string(
          output_mosaic.center)
      print(f'Mosaic has name {output_mosaic.name} and id {output_mosaic.id}')

      output_mosaics.append(output_mosaic)
    except Exception as e:
      traceback.print_exc()

  return output_mosaics
