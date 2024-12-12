from spherical_geometry.polygon import SphericalPolygon, SingleSphericalPolygon  # type: ignore
import astropy.io.fits as pyfits  # type: ignore
from astropy.wcs import WCS  # type: ignore
import ray
import numpy as np
from dataclasses import fields
from typing import Any

from context import context
from clustering import find_groups
from catalogs import position_string
from targets import Target
from targets.types import TargetHash
import geometry.footprints as footprints
from hashing import hash_list
from stacks.composite_stack import CompositeStack
from stacks.astrometric_attributes import AstrometricAttributes
from profiling import track_performance
from stacks.types import StackHash


@ray.remote
@track_performance
def group_stacks_into_targets(
    stacks: dict[StackHash, CompositeStack]) -> dict[TargetHash, Target]:
  metadata = context.metadata

  cosine_similarity_threshold = context.cosine_similarity_threshold

  def get_spherical_polygon(wcs: WCS) -> SphericalPolygon:
    footprint = SingleSphericalPolygon.from_wcs(wcs)
    if not footprint.is_clockwise():
      footprint = SingleSphericalPolygon(footprint.points[::-1])
    assert not SphericalPolygon.self_intersect(footprint.points)
    return footprint

  def center_polygon_image_scale(
      stack: CompositeStack) -> tuple[np.ndarray, SphericalPolygon, float]:
    first_frame = stack.all_frames()[0]
    plate_solution = metadata[first_frame].plate_solution
    return (np.asarray(footprints.radec_to_vector(*stack.center)),
            get_spherical_polygon(pyfits.Header(plate_solution.header)),
            stack.image_scale)

  values = {
      id: center_polygon_image_scale(stack)
      for id, stack in stacks.items()
      if type(stack) is CompositeStack and stack.image_type == 'light'
  }

  print(f'### Finding targets out of {len(values)} light-frame stacks')

  # TODO: Make this IoU threashold a knob in the UI.
  # TODO: Don't combine stacks with different image scales into the same
  # target.
  def comparison_fn(a: tuple[np.ndarray, SphericalPolygon, float],
                    b: tuple[np.ndarray, SphericalPolygon, float]) -> bool:
    a_center, a_polygon, a_image_scale = a
    b_center, b_polygon, b_image_scale = b

    image_scales = a_image_scale, b_image_scale
    if not np.allclose(image_scales, np.mean(image_scales), atol=0.1):
      return False

    if np.dot(a_center, b_center) < cosine_similarity_threshold:
      return False

    iou = a_polygon.intersection(b_polygon).area() / a_polygon.union(
        b_polygon).area()
    return iou > 0.6

  groups = find_groups(values, comparison_fn, min_component_members=1)

  def new_target(stack_ids: list[StackHash]) -> Target:
    # TODO: This just uses the first stack's footprint, which is very simple,
    # but maybe not very precise. The more precise implementation is here in
    # a comment.
    # footprint = next(
    #     SphericalPolygon.multi_intersection([
    #         SphericalPolygon(stacks[stack_id].footprint)
    #         for stack_id in stack_ids
    #     ]).points).tolist()
    # projected_footprint = footprints.all_projections(
    #     SphericalPolygon(footprint))

    # TODO: Validate all stacks here.

    stable_hash = TargetHash(stacks[sorted(
        stack_ids, key=lambda s: stacks[s].oldest_date)[0]].stable_hash)
    content_hash = hash_list(stacks[s].content_hash for s in stack_ids)

    target_attributes: dict[str, Any] = {
        'stable_hash': stable_hash,
        'oldest_date': min(stacks[s].oldest_date for s in stack_ids),
        'newest_date': min(stacks[s].newest_date for s in stack_ids),
        'content_hash': content_hash,
        'stack_ids': stack_ids
    }

    astrometric_attributes: dict[str, Any] = {
        f.name: getattr(stacks[stack_ids[0]], f.name)
        for f in fields(AstrometricAttributes)
    }

    attributes = target_attributes | astrometric_attributes

    return Target(**attributes)

  targets_list = [new_target(g) for g in groups]
  targets = {t.stable_hash: t for t in targets_list}

  print(f'### Found {len(targets)} targets: {targets.keys()}')

  # TODO: Computing the result and storing the result are two different
  # responsibilities. Perhaps separate patterns like this?
  context.or_equals('grouped_targets', targets)

  return targets
