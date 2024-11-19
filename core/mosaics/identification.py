import traceback
import structlog
from spherical_geometry.polygon import SphericalPolygon, SingleSphericalPolygon  # type: ignore
from spherical_geometry import great_circle_arc
import ray
import numpy as np
import random
from dataclasses import asdict
from enum import Enum

from context import context
from mosaics import Mosaic
from mosaics.types import MosaicHash
from catalogs import position_string
from geometry import footprints
from hashing import hash_list
import units
from profiling import track_performance
from targets.types import TargetHash
from targets.target import Target
from mosaics.debugging import plot_mosaic

logger = structlog.get_logger()
log = logger


class Overlap(Enum):
  INCOMPATIBLE = 1
  LONG_EDGE = 2
  SHORT_EDGE = 3


def targets_compatible_overlap(a: Target,
                               b: Target,
                               debug: bool = False) -> bool:
  if a.stable_hash == b.stable_hash:
    return Overlap.INCOMPATIBLE

  if not a.footprint.intersects_poly(b.footprint):
    return Overlap.INCOMPATIBLE

  # TODO: Don't bother checking intersection of targets that are very far
  # apart on the sky?

  if debug:
    print(f'### targets_compatible_overlap {a.stable_hash} and {b.stable_hash}')
    print(f'a.stable_hash', a.stable_hash)
    print(f'b.stable_hash', b.stable_hash)
    print(f'a.rotation_angle {np.degrees(a.rotation_angle)} degrees')
    print(f'b.rotation_angle {np.degrees(b.rotation_angle)} degrees')

  # TODO: This essentially forces mosaics to be made with an equatorial mount.
  # if (a.rotation_angle - b.rotation_angle) > 0.17:
  #   return False

  intersects = a.footprint.intersects_poly(b.footprint)

  if debug:
    print(f'a.area {a.footprint.area()}')
    print(f'b.area {b.footprint.area()}')
    print(f'intersects {intersects}')

  # Returns a tuple of (short edge, long edge).
  def compute_edge_lengths(p: SphericalPolygon) -> tuple[float, float]:
    min_length, max_length = np.inf, -np.inf
    points = list(p.points)

    for i in range(len(points) - 1):
      length = great_circle_arc.length(points[i], points[i + 1])
      min_length = min(min_length, length)
      max_length = max(max_length, length)

    return min_length, max_length

  frame_short_edge, frame_long_edge = compute_edge_lengths(a.footprint)

  if debug:
    print(f'frame_short_edge {frame_short_edge}')
    print(f'frame_long_edge {frame_long_edge}')

  # Compute the length of the long and short edges of the intersection of the
  # two polygons.
  intersection_polygons = a.footprint.intersection(b.footprint).polygons
  if len(intersection_polygons) != 1:
    if debug:
      print(f'targets {a.stable_hash} and {b.stable_hash} do not intersect')
    return Overlap.INCOMPATIBLE

  intersection = SingleSphericalPolygon.convex_hull(
      intersection_polygons[0].points)
  if intersection == None:
    print(f'could not find rectangular convex hull of intersection')
    return Overlap.INCOMPATIBLE

  intersection_short_edge, intersection_long_edge = compute_edge_lengths(
      intersection)
  a_fraction = intersection.area() / a.footprint.area()
  b_fraction = intersection.area() / b.footprint.area()

  if debug:
    print(f'intersection_short_edge {intersection_short_edge}')
    print(f'intersection_long_edge {intersection_long_edge}')
    print(f'intersection / a area {a_fraction}')
    print(f'intersection / b area {b_fraction}')

  if a_fraction < 0.1 or a_fraction > 0.3:
    return Overlap.INCOMPATIBLE
  if b_fraction < 0.1 or b_fraction > 0.3:
    return Overlap.INCOMPATIBLE

  aspect_ratio = intersection_long_edge / intersection_short_edge

  if debug:
    print(f'aspect_ratio {aspect_ratio}')

  if aspect_ratio < 3:
    return Overlap.INCOMPATIBLE

  # To figure out whether the pair is oriented horizontally or vertically,
  # look at the longest edge of the intersection. If it roughly matches the
  # short edge, then the pair is oriented vertically, joined by their short
  # edges.
  def error(a, b):
    return abs(a - b) / a

  long_edge_error = error(frame_long_edge, intersection_long_edge)
  short_edge_error = error(frame_short_edge, intersection_long_edge)

  if long_edge_error > 0.2 and short_edge_error > 0.2:
    if debug:
      print(f'targets {a.stable_hash} and {b.stable_hash} '
            f'have unclear orientation long_edge_error {long_edge_error} '
            f'short_edge_error {short_edge_error}')
    return Overlap.INCOMPATIBLE

  orientation = (Overlap.LONG_EDGE
                 if long_edge_error < short_edge_error else Overlap.SHORT_EDGE)
  if debug:
    print(f'long_edge_error {long_edge_error}')
    print(f'short_edge_error {short_edge_error}')
    print(f'orientation {orientation}')
    print(f'targets are compatible with orientation {orientation}')

  return orientation


def find_strips(target_ids: list[TargetHash],
                debug: bool = False) -> list[list[TargetHash]]:
  targets = context.grouped_targets

  strips = []
  while target_ids:
    strip = [target_ids[0]]
    for target_id in target_ids:
      if any(
          targets_compatible_overlap(targets[target_id], targets[tid]) ==
          Overlap.LONG_EDGE for tid in strip):
        strip.append(target_id)

    strips.append(strip)

    target_ids = [t for t in target_ids if t not in strip]

  def order_strip(strip):
    strip = list(sorted(strip, key=lambda t: targets[t].center[0]))
    end_target_id = None
    for target_id in strip:
      if sum(1 for tid in strip if targets_compatible_overlap(
          targets[target_id], targets[tid]) != Overlap.INCOMPATIBLE) == 1:
        end_target_id = target_id
        break

    assert end_target_id

    output_strip = [end_target_id]
    while len(output_strip) < len(strip):
      for target_id in strip:
        if targets_compatible_overlap(
            targets[output_strip[-1]],
            targets[target_id]) != Overlap.INCOMPATIBLE:
          output_strip.append(target_id)

    if debug:
      plot_mosaic(output_strip, title=f'Output strip #{len(strips)}')

    return output_strip

  output_strips = []
  for strip in strips:
    ordered_strip = order_strip(strip)
    if debug:
      print(f'### before ordering, strip is {strip}')
      print(f'### after ordering, strip is {ordered_strip}')
    output_strips.append(ordered_strip)

  return output_strips


def _identify_overlapping_targets() -> list[list[TargetHash]]:
  stacks = context.grouped_stacks
  targets = context.grouped_targets

  if not targets:
    return []

  def roughly_same_area(a: SphericalPolygon, b: SphericalPolygon) -> bool:
    ratio = a.area() / b.area()
    return ratio > 0.9 and ratio < 1.1

  # TODO: Targets should be really be checked that they overlap mostly along
  # one axis (e.g. the long edge or the short edge), within some amount (say
  # 5-30%).

  # Sort the targets by their center RA.
  target_ids = list(sorted(targets.keys(), key=lambda t: targets[t].center[0]))
  log.info('Identifying overlapping targets', num_targets=len(targets))

  mosaics: list[list[TargetHash]] = []
  for target_id in target_ids:
    matching_mosaic_indices = []
    for i, mosaic in enumerate(mosaics):
      compatible_area = roughly_same_area(targets[target_id].footprint,
                                          targets[mosaic[0]].footprint)
      compatible_overlap = any(
          targets_compatible_overlap(targets[target_id], targets[t]) !=
          Overlap.INCOMPATIBLE for t in mosaic)
      mosaic_image_scales = [
          stacks[s].image_scale for t in mosaic for s in targets[t].stack_ids
      ]
      # TODO: Image scale is maybe not a requirement for mosaicking. It's a
      # complex feature that requires complex UI, so leave it out for now.
      target_image_scales = [
          stacks[s].image_scale for s in targets[target_id].stack_ids
      ]
      assert np.allclose(target_image_scales,
                         np.mean(target_image_scales),
                         atol=0.1)
      compatible_image_scales = np.allclose(mosaic_image_scales,
                                            np.mean(target_image_scales),
                                            atol=0.1)

      if compatible_area and compatible_overlap and compatible_image_scales:
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
  for mosaic in mosaics:
    print('### Group of overlapping targets:', mosaic)

  return mosaics


@ray.remote
@track_performance
def identify_mosaics(debug: bool = False) -> dict[MosaicHash, Mosaic]:
  targets = context.grouped_targets
  print(f'### Identifying mosaics from {len(targets)} targets')

  mosaics = _identify_overlapping_targets()
  if not mosaics:
    print('### No overlapping targets found')
    context.grouped_mosaics = {}
    return {}

  print(f'Found {len(mosaics)} mosaics')

  output_mosaics = {}
  for mosaic in mosaics:
    assert mosaic
    mosaic = list(sorted(mosaic, key=lambda t: targets[t].center[0]))

    if debug:
      plot_mosaic(mosaic, title=f'Mosaic')

    try:
      # TODO: Validate all targets here.

      stable_hash = targets[sorted(
          mosaic, key=lambda t: targets[t].oldest_date)[0]].stable_hash
      content_hash = hash_list(targets[t].content_hash for t in mosaic)

      oldest_date = min(targets[t].oldest_date for t in mosaic)
      newest_date = min(targets[t].newest_date for t in mosaic)

      centers = [targets[t].center for t in mosaic]
      mean_center = np.mean(centers, axis=0)
      center = list(mean_center)
      name = position_string.canonical_target_string(center)

      # This doesn't seem work for polygons with holes in them.
      # TODO: Consider using multi_union but falling back to convex hull when
      # it doesn't work?
      # old_footprint = next(
      #     SphericalPolygon.multi_union([
      #         SphericalPolygon(targets[target_id].footprint)
      #         for target_id in mosaic
      #     ]).points).tolist()

      all_points = np.concatenate(
          [targets[target_id].footprint.points for target_id in mosaic])

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
      projected_footprints = footprints.all_projections(footprint)

      strips = find_strips(mosaic)
      print('### strips', strips)
      target_ids = []
      tile_indices = []
      for i, strip in enumerate(strips):
        for j, target_id in enumerate(strip):
          target_ids.append(target_id)
          tile_indices.append((i, j))

      output_mosaic = Mosaic(
          name=name,
          stable_hash=stable_hash,
          content_hash=content_hash,
          target_ids=target_ids,
          tile_indices=tile_indices,
          #  origin=result.origin,
          origin=np.asarray(0),
          # basis_vectors=result.basis_vectors,
          basis_vectors=np.asarray(0),
          oldest_date=oldest_date,
          newest_date=newest_date,
          center=center,
          footprint=footprint,
          projected_footprints=projected_footprints)
      print(f'### Made output_mosaic {output_mosaic}')

      # print(f'Mosaic has name {name} stable_hash {stable_hash} and '
      #       f'basis_vectors {result.basis_vectors}')
      output_mosaics[stable_hash] = output_mosaic
    except Exception as e:
      traceback.print_exc()

  # TODO: Computing the result and storing the result are two different
  # responsibilities. Perhaps separate patterns like this?
  context.grouped_mosaics = output_mosaics
  return output_mosaics
