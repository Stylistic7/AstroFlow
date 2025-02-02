import traceback
import structlog
from spherical_geometry.polygon import SphericalPolygon, SingleSphericalPolygon  # type: ignore
from spherical_geometry.vector import vector_to_lonlat
from spherical_geometry import great_circle_arc
import ray
import numpy as np
import random
from enum import Enum
from typing import Any
from dataclasses import asdict

from context import context
from mosaics.types import Mosaic, MosaicHash, MosaicsDict
from catalogs import position_string
from geometry import footprints
from hashing import hash_list
from targets.types import TargetHash, Target, TargetsDict
from mosaics.debugging import plot_mosaic
from dependency_graph.astroflow_node import AstroFlowNode
from stacks.astrometric_attributes import AstrometricAttributes
from stacks.chronology_attributes import ChronologyAttributes

# TODO: Make the functions in this file functional (e.g. pass TargetsDict into
# them, vs. relying on context).

logger = structlog.get_logger()
log = logger


class Overlap(Enum):
  IDENTICAL = 1
  NON_OVERLAPPING = 2
  INCOMPATIBLE = 3
  LONG_EDGE = 4
  SHORT_EDGE = 5
  CORNER = 6


def roughly_same_area(a: SingleSphericalPolygon,
                      b: SingleSphericalPolygon) -> bool:
  ratio = a.area() / b.area()
  return ratio > 0.9 and ratio < 1.1


def roughly_same_image_scale(a: float, b: float) -> bool:
  ratio = a / b
  return ratio > 0.9 and ratio < 1.1


@ray.remote
def target_compatible_with_mosaic(target_id: TargetHash,
                                  mosaic: set[TargetHash],
                                  allowed_overlaps,
                                  debug: bool = False) -> bool:
  targets = context.grouped_targets
  target = targets[target_id]

  # The two mosaics can be merged together if:
  #
  # - all of the targets have compatible image scales and areas
  # - at least one target in `a` overlaps LONG_EDGE or SHORT_EDGE with a
  #   a target in `b`.
  # - none of the targets in `a` overlap INCOMPATIBLE with any targets in `b`.
  num_good_overlaps = 0
  for tid in mosaic:
    if not roughly_same_image_scale(target.image_scale,
                                    targets[tid].image_scale):
      return False

    if not roughly_same_area(target.footprint, targets[tid].footprint):
      return False

    overlap = targets_overlap(target, targets[tid], debug)
    if overlap is Overlap.INCOMPATIBLE:
      return False

    if overlap in allowed_overlaps:
      num_good_overlaps += 1

  return num_good_overlaps > 0


def targets_overlap(a: Target, b: Target, debug: bool = False) -> bool:
  if a.stable_hash == b.stable_hash:
    return Overlap.IDENTICAL

  if debug:
    print(f'### targets_overlap {a.stable_hash} and {b.stable_hash}')

  if not a.footprint.intersects_poly(b.footprint):
    if debug:
      print(f'targets {a.stable_hash} and {b.stable_hash} do not intersect')
    return Overlap.NON_OVERLAPPING

  # TODO: Don't bother checking intersection of targets that are very far
  # apart on the sky?

  if debug:
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
  if len(intersection_polygons) == 0:
    return Overlap.NON_OVERLAPPING

  if len(intersection_polygons) > 1:
    if debug:
      print(f'targets {a.stable_hash} and {b.stable_hash}'
            f'have {len(intersection_polygons)} intersection polygons')
      return Overlap.INCOMPATIBLE

  intersection = SingleSphericalPolygon.convex_hull(
      intersection_polygons[0].points)
  if intersection == None:
    print(f'could not find rectangular convex hull of intersection')
    return Overlap.INCOMPATIBLE

  if intersection.area() > 12:
    if debug:
      print(
          f'### intersection had to be inverted, area is {intersection.area()}')
    intersection = intersection.invert_polygon()

  intersection_short_edge, intersection_long_edge = compute_edge_lengths(
      intersection)
  a_fraction = intersection.area() / a.footprint.area()
  b_fraction = intersection.area() / b.footprint.area()

  long_edge_fraction = intersection_long_edge / frame_long_edge
  short_edge_fraction = intersection_short_edge / frame_short_edge

  if debug:
    print(f'intersection.area() {intersection.area()}')
    print(f'intersection_short_edge {intersection_short_edge}')
    print(f'intersection_long_edge {intersection_long_edge}')
    print(f'long_edge_fraction {long_edge_fraction}')
    print(f'short_edge_fraction {short_edge_fraction}')
    print(f'intersection aspect ratio '
          f'{intersection_long_edge / intersection_short_edge}')
    print(f'intersection / a area {a_fraction}')
    print(f'intersection / b area {b_fraction}')

  # For orientation LONG_EDGE, the intersection's long edge is similar to the
  # frame's long edge. The intersection's short edge is much shorter than
  # the frame's short edge.
  #
  # For orientation SHORT_EDGE, the intersection's long edge is similar to the
  # frame's short edge. The intersection's short edge is much shorter than
  # the frame's long edge.
  def error(a, b):
    return abs(a - b) / a

  if error(intersection_long_edge, frame_long_edge) < 0.25 and (
      intersection_short_edge / frame_short_edge) < 0.3:
    orientation = Overlap.LONG_EDGE
  elif error(intersection_long_edge, frame_short_edge) < 0.25 and (
      intersection_short_edge / frame_long_edge) < 0.3:
    orientation = Overlap.SHORT_EDGE
  elif (intersection_long_edge / frame_long_edge) < 0.3 and (
      intersection_short_edge / frame_short_edge) < 0.3:
    orientation = Overlap.CORNER
  else:
    if debug:
      print(f'targets {a.stable_hash} and {b.stable_hash} '
            f'have unclear orientation')
    return Overlap.INCOMPATIBLE

  if debug:
    print(f'orientation {orientation}')
    print(f'targets are compatible with orientation {orientation}')

  return orientation


def group_targets(
    target_ids: list[TargetHash],
    allowed_overlaps,
    debug: bool = False,
) -> list[set[TargetHash]]:
  mosaics = []
  for target_id in target_ids:
    # Find mosaics that are compatible with the current target, in parallel.
    results = ray.get([
        target_compatible_with_mosaic.remote(target_id, mosaic,
                                             allowed_overlaps, debug)
        for mosaic in mosaics
    ])
    compatible_mosaics = [
        mosaic for mosaic, match in zip(mosaics, results) if match
    ]

    if compatible_mosaics:
      if debug:
        print(f'### target {target_id} will be merged with '
              f'existing mosaics {compatible_mosaics}')
      merged_mosaic = {target_id}
      for mosaic in compatible_mosaics:
        merged_mosaic = merged_mosaic.union(mosaic)
      mosaics = [c for c in mosaics if c not in compatible_mosaics]
      mosaics.append(merged_mosaic)
    else:
      mosaics.append({target_id})

  return mosaics


def find_strips(target_ids: list[TargetHash],
                debug: bool = False) -> list[list[TargetHash]]:
  targets = context.grouped_targets

  target_ids = list(sorted(target_ids, key=lambda t: targets[t].center[0]))

  strips = group_targets(target_ids, allowed_overlaps=[Overlap.LONG_EDGE])
  if debug:
    for i, strip in enumerate(strips):
      print(f'### Strip {i} is {strip}')

  # TODO: Maybe use this same algorithm to order the strips themselves to make
  # sure the output strips can be merged with PhotometricMergeMosaic.
  def order_strip(strip):
    strip_length = len(strip)
    strip = list(sorted(strip, key=lambda t: targets[t].center[0]))

    if len(strip) < 3:
      return strip

    end_target_id = None
    for target_id in strip:
      overlap_count = sum(1 for tid in strip if targets_overlap(
          targets[target_id], targets[tid]) == Overlap.LONG_EDGE)
      if debug:
        print(f'### target {target_id} has overlap_count {overlap_count}')
      if overlap_count == 1:
        if debug:
          print(f'### chose {target_id} for the end of the strip')
        end_target_id = target_id
        break

    assert end_target_id

    output_strip = [end_target_id]
    strip.remove(end_target_id)
    while len(output_strip) < strip_length:
      for target_id in strip:
        if targets_overlap(targets[output_strip[-1]],
                           targets[target_id]) == Overlap.LONG_EDGE:
          if debug:
            print(f'### added {target_id} to strip')
          output_strip.append(target_id)
          strip.remove(target_id)

    assert len(output_strip) == strip_length
    return output_strip

  output_strips = []
  for strip in strips:
    ordered_strip = order_strip(strip)
    if debug:
      print(f'### before ordering, strip is {strip}')
      print(f'### after ordering, strip is {ordered_strip}')
    output_strips.append(ordered_strip)

  return output_strips


def _identify_overlapping_targets(
    debug: bool = False) -> list[list[TargetHash]]:
  targets = context.grouped_targets
  if not targets:
    return []

  # Sort the targets by their center RA.
  target_ids = list(sorted(targets.keys(), key=lambda t: targets[t].center[0]))
  log.info('Identifying overlapping targets', num_targets=len(targets))

  mosaics = group_targets(
      target_ids,
      allowed_overlaps=[Overlap.LONG_EDGE, Overlap.SHORT_EDGE],
      debug=debug)

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


@AstroFlowNode()
def identify_mosaics(targets: TargetsDict,
                     debug: bool = False) -> dict[MosaicHash, Mosaic]:
  print(f'### Identifying mosaics from {len(targets)} targets')

  mosaics = _identify_overlapping_targets(debug)
  if not mosaics:
    print('### No overlapping targets found')
    return MosaicsDict()

  print(f'Found {len(mosaics)} mosaics')

  output_mosaics = {}
  for mosaic in mosaics:
    assert mosaic
    mosaic = list(sorted(mosaic, key=lambda t: targets[t].center[0]))

    try:
      # TODO: Validate all targets here.

      stable_hash = targets[sorted(
          mosaic, key=lambda t: targets[t].oldest_date)[0]].stable_hash
      content_hash = hash_list(targets[t].content_hash for t in mosaic)

      oldest_date = min(targets[t].oldest_date for t in mosaic)
      newest_date = min(targets[t].newest_date for t in mosaic)

      image_scale = np.mean([targets[t].image_scale for t in mosaic])
      rotation_angle = np.mean([targets[t].rotation_angle for t in mosaic])

      # This doesn't seem work for polygons with holes in them.
      # TODO: Consider using multi_union but falling back to convex hull when
      # it doesn't work?
      # old_footprint = next(
      #     SphericalPolygon.multi_union([
      #         SphericalPolygon(targets[target_id].footprint)
      #         for target_id in mosaic
      #     ]).points).tolist()

      strips = find_strips(mosaic, debug)
      print('### strips', strips)
      target_ids = []
      tile_indices = []
      for i, strip in enumerate(strips):
        for j, target_id in enumerate(strip):
          target_ids.append(target_id)
          tile_indices.append((i, j))

      for target_id in mosaic:
        print(f'### Target {target_id} '
              f'has {len(targets[target_id].footprint.points)} points')

      all_points = [
          targets[target_id].footprint.points[:-1] for target_id in mosaic
      ]
      all_points = np.concatenate(all_points)
      print(f'### Mosaic composed of {len(mosaic)} targets '
            f'and {all_points.shape} points')
      footprint = SingleSphericalPolygon.convex_hull(all_points)

      # TODO: This is turbo weird, but appears to work. Just randomize the
      # points until the `convex_hull()` operation is successful.
      for i in range(10):
        if SphericalPolygon.self_intersect(footprint.points):
          points = footprint.points
          random.shuffle(points)
          footprint = SingleSphericalPolygon.convex_hull(points)
        else:
          print(f'### convex_hull successful after {i} iterations '
                f'with {len(footprint.points)} points')
          break

      # assert not SphericalPolygon.self_intersect(footprint.points)
      projected_footprints = footprints.all_projections(footprint)

      center = vector_to_lonlat(*np.mean(footprint.points, axis=0))
      name = position_string.canonical_target_string(center)

      chronology_attributes = ChronologyAttributes(oldest_date, newest_date)
      astrometric_attributes = AstrometricAttributes(
          name=name,
          center=list(center),
          image_scale=image_scale,
          rotation_angle=rotation_angle,
          footprint=footprint,
          projected_footprints=projected_footprints)

      attributes: dict[str, Any] = asdict(chronology_attributes) | asdict(
          astrometric_attributes) | {
              'stable_hash': stable_hash,
              'content_hash': content_hash,
              'target_ids': target_ids,
              'tile_indices': tile_indices,
              'strips': strips,
          }

      output_mosaic = Mosaic(**attributes)
      print(f'### Made output_mosaic {output_mosaic}')

      if debug:
        plot_mosaic(output_mosaic, title=f'Mosaic')

      output_mosaics[stable_hash] = output_mosaic
    except Exception as e:
      traceback.print_exc()

  return output_mosaics
