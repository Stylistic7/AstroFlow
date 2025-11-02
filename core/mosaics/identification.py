import traceback
import structlog
from spherical_geometry.polygon import SphericalPolygon, SingleSphericalPolygon  # type: ignore
from spherical_geometry.vector import vector_to_lonlat
from spherical_geometry import great_circle_arc
import ray
import numpy as np
import random
from typing import Any, Sequence
from dataclasses import asdict
from beartype import beartype
import time
from collections import defaultdict
from itertools import combinations

from context import context
from mosaics.types import Mosaic, MosaicHash, MosaicsDict, Strip
from catalogs import position_string
from geometry import footprints
from hashing import hash_list
from targets.types import TargetHash, Target, TargetsDict
from mosaics.debugging import plot_mosaic
from mosaics.disjoint_set import HashDisjointSet
from mosaics.overlap import Overlap
from decorators import cached
from attributes import AstrometricAttributes, ChronologyAttributes
from attributes.astrometric_attributes import healpix_cells_from_poly

# TODO: Make the functions in this file functional (e.g. pass TargetsDict into
# them, vs. relying on context).

logger = structlog.get_logger()
log = logger


def roughly_same_area(a: SingleSphericalPolygon,
                      b: SingleSphericalPolygon) -> bool:
  ratio = a.area() / b.area()
  return ratio > 0.9 and ratio < 1.1


def roughly_same_image_scale(a: float, b: float) -> bool:
  ratio = a / b
  return ratio > 0.9 and ratio < 1.1


@cached.remote
def targets_overlap(a: Target,
                    b: Target,
                    min_area_frac=0.03,
                    span_match=0.5,
                    thick_min=0.05,
                    thick_max=0.3) -> Overlap:
  if a is b:
    return Overlap.IDENTICAL

  if not roughly_same_image_scale(a.image_scale, b.image_scale):
    return Overlap.INCOMPATIBLE

  a, b = a.footprint, b.footprint
  if not roughly_same_area(a, b):
      return Overlap.INCOMPATIBLE

  if not a.intersects_poly(b):
    return Overlap.NON_OVERLAPPING

  inter = a.intersection(b)
  polys = getattr(inter, "polygons", [inter])
  if not polys:
    return Overlap.NON_OVERLAPPING
  inter = max(polys, key=lambda p: p.area())
  if inter.area() / max(min(a.area(), b.area()), 1e-30) < min_area_frac:
    return Overlap.INCOMPATIBLE

  def lonlat(p):
    xyz = np.array(list(p.points))
    return np.c_[np.arctan2(xyz[:, 1], xyz[:, 0]), np.arcsin(xyz[:, 2])]

  def gnomonic(ll, lon0, lat0):
    lo, la = ll[:, 0], ll[:, 1]
    cl0, sl0 = np.cos(lat0), np.sin(lat0)
    co0, so0 = np.cos(lon0), np.sin(lon0)
    cl, sl = np.cos(la), np.sin(la)
    co, so = np.cos(lo), np.sin(lo)
    cosc = sl0 * sl + cl0 * cl * (co * co0 + so * so0)
    x = cl * (so * co0 - co * so0) / cosc
    y = (cl0 * sl - sl0 * cl * (co * co0 + so * so0)) / cosc
    return np.c_[x, y]

  def obb(xy):
    X = xy - xy.mean(0)
    cov = (X.T @ X) / max(len(X) - 1, 1)
    v, e = np.linalg.eigh(cov)
    proj = X @ e
    lo, hi = proj.min(0), proj.max(0)
    w, h = (hi - lo)
    S, L = sorted((w, h))
    return S, L  # minor/major lengths in the plane

  # tangent plane at intersection centroid (symmetric)
  xyz = np.array(list(inter.points))
  c = xyz.mean(0)
  c /= np.linalg.norm(c)
  lon0, lat0 = np.arctan2(c[1], c[0]), np.arcsin(c[2])
  proj = lambda p: gnomonic(lonlat(p), lon0, lat0)

  S_a, L_a = obb(proj(a))
  S_b, L_b = obb(proj(b))
  S_i, L_i = obb(proj(inter))
  span, thick = (L_i, S_i) if L_i >= S_i else (S_i, L_i)

  # symmetric scores: use a frame only if thickness test passes
  def score_long(S,
                 L):  # span vs long edge; thickness vs that frame's short edge
    t = thick / S
    return (span / L) if (thick_min <= t <= thick_max) else 0.0

  def score_short(S,
                  L):  # span vs short edge; thickness vs that frame's long edge
    t = thick / L
    return (span / S) if (thick_min <= t <= thick_max) else 0.0

  long_score = max(score_long(S_a, L_a), score_long(S_b, L_b))
  short_score = max(score_short(S_a, L_a), score_short(S_b, L_b))

  if long_score >= span_match and short_score < span_match:
    return Overlap.LONG_EDGE
  if short_score >= span_match and long_score < span_match:
    return Overlap.SHORT_EDGE
  if long_score >= span_match and short_score >= span_match:
    if long_score > short_score:
      return Overlap.LONG_EDGE
    if short_score > long_score:
      return Overlap.SHORT_EDGE
  return Overlap.INCOMPATIBLE


def group_targets(
    target_ids: list[TargetHash],
    allowed_overlaps,
    debug: bool = False,
) -> list[set[TargetHash]]:
  assert len(target_ids) > 0
  targets = context.grouped_targets

  target_ids_for_cell = defaultdict(list)
  for target_id in target_ids:
    target = targets[target_id]
    assert len(target.healpix_cells) > 0
    for cell in target.healpix_cells:
      target_ids_for_cell[cell].append(target_id)

  assert len(target_ids_for_cell) > 0

  disjoint_set = HashDisjointSet()
  # TODO: Parallelize these.
  for cell, tids in target_ids_for_cell.items():
    if len(tids) < 2:
      continue

    for a, b in combinations(sorted(tids), 2):
      overlap = ray.get(targets_overlap.remote(targets[a], targets[b]))
      if overlap in allowed_overlaps:
        print(f'### union {a=} {b=} with overlap {overlap}')
        disjoint_set.union(a, b)

  mosaics = disjoint_set.groups()

  singletons = [[t] for t in target_ids if not any(t in m for m in mosaics)]

  result = mosaics + singletons
  assert sum(len(r) for r in result) == len(target_ids)
  return result


def test_commutativity():
  targets = context.grouped_targets

  for a in targets.keys():
    for b in targets.keys():
      a_to_b = ray.get(targets_overlap.remote(targets[a], targets[b]))
      b_to_a = ray.get(targets_overlap.remote(targets[b], targets[a]))
      assert a_to_b == b_to_a, f"### {a_to_b=} {b_to_a=}"


def find_strips(target_ids: list[TargetHash],
                debug: bool = False) -> list[Strip]:
  targets = context.grouped_targets

  strips = group_targets(target_ids, allowed_overlaps=[Overlap.LONG_EDGE])
  if debug:
    for i, strip in enumerate(strips):
      print(f'### Strip {i} is {strip}')

  # TODO: Maybe use this same algorithm to order the strips themselves to make
  # sure the output strips can be merged with PhotometricMergeMosaic.
  @beartype
  def order_strip(strip: set[TargetHash]) -> Strip:
    strip_length = len(strip)
    strip = list(sorted(strip, key=lambda t: targets[t].center[0]))

    if len(strip) < 3:
      return strip

    def overlap_count(target_id):
      return sum(
          1 for tid in strip
          if ray.get(targets_overlap.remote(targets[target_id],
                                            targets[tid])) == Overlap.LONG_EDGE)

    overlap_counts = {tid: overlap_count(tid) for tid in strip}
    print(f'### overlap_counts {overlap_counts}')
    assert set(overlap_counts.values()) == {1, 2}
    ones = [k for k, v in overlap_counts.items() if v == 1]
    assert len(ones) == 2
    end_target_id = ones[0]

    output_strip = [end_target_id]
    strip.remove(end_target_id)
    while len(output_strip) < strip_length:
      print(f'### order_strip len(output_strip) {len(output_strip)}')
      starting_length = len(output_strip)
      for target_id in strip:
        overlap = ray.get(
            targets_overlap.remote(targets[output_strip[-1]],
                                   targets[target_id]))
        if overlap == Overlap.LONG_EDGE:
          if debug:
            print(f'### added {target_id} to strip')
          output_strip.append(target_id)
          strip.remove(target_id)
          break
        else:
          print(f'### end of strip target {output_strip[-1]} '
                f'has overlap {overlap} with target {target_id}')

      assert len(output_strip
                ) != starting_length, "### order_strip did not make progress"

    assert len(output_strip) == strip_length
    return Strip(output_strip)

  output_strips = []
  for i, strip in enumerate(strips):
    print(f'### Ordering strip {i} / {len(strips)}')
    ordered_strip = order_strip(set(strip))
    if debug:
      print(f'### before ordering, strip is {strip}')
      print(f'### after ordering, strip is {ordered_strip}')
    output_strips.append(ordered_strip)

  # TODO: This is a bit of a hack. Consider more principled ways of sorting
  # strips.
  def min_dec(strip: list[TargetHash]):
    return min(targets[t].center[1] for t in strip)

  # Sort each strip individually by increasing RA.
  strips = [sorted(s, key=lambda t: targets[t].center[0]) for s in strips]

  # Sort the strips by increasing Dec.
  return sorted(strips, key=min_dec)


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

  mosaics = [sorted(m) for m in mosaics if len(m) > 1]
  for mosaic in mosaics:
    print('### Group of overlapping targets:', mosaic)

  return mosaics


@cached.remote
def identify_mosaics(targets: TargetsDict,
                     debug: bool = True) -> dict[MosaicHash, Mosaic]:
  print(f'### Identifying mosaics from {len(targets)} targets')

  # test_commutativity()

  mosaics = _identify_overlapping_targets(debug)
  if not mosaics:
    print('### No overlapping targets found')
    return MosaicsDict()

  print(f'Found {len(mosaics)} mosaics')

  # TODO: Parallelize these
  output_mosaics = {}
  for i, mosaic in enumerate(mosaics):
    print(f'### Processing mosaic {i} / {len(mosaics)} '
          f'with {len(mosaic)} targets: {mosaic}')

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

      healpix_cells = healpix_cells_from_poly(footprint)

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

      chronology_attributes = ChronologyAttributes(stable_hash, oldest_date,
                                                   newest_date)
      astrometric_attributes = AstrometricAttributes(
          name=name,
          center=list(center),
          image_scale=image_scale,
          rotation_angle=rotation_angle,
          footprint=footprint,
          healpix_cells=healpix_cells,
          projected_footprints=projected_footprints)

      attributes: dict[str, Any] = asdict(chronology_attributes) | asdict(
          astrometric_attributes) | {
              'stable_hash': stable_hash,
              'content_hash': content_hash,
              'target_ids': sorted(mosaic),
              'strips': strips,
          }

      output_mosaic = Mosaic(**attributes)
      print(f'### Made output_mosaic {output_mosaic}')

      # if debug:
      #   plot_mosaic(output_mosaic, title=f'Mosaic')

      output_mosaics[stable_hash] = output_mosaic
    except Exception as e:
      traceback.print_exc()

  return output_mosaics
