from spherical_geometry.polygon import SphericalPolygon
import astropy.io.fits as pyfits
import ray
import numpy as np

import file_io
from new_context import new_context
import spatial_association
import position_string
from target import Target
import ray_diskcache
import footprints


@ray.remote
def group_stacks_into_targets():
  stacks = new_context.stacks
  plate_solutions = new_context.plate_solutions

  values = {
      id: (np.asarray(footprints.radec_to_vector(*stack.center)),
           SphericalPolygon.from_wcs(
               pyfits.Header(
                   plate_solutions[stack.all_frames(stacks)[0]].header)))
      for id, stack in stacks.items()
      if stack.geometric_hash and stack.image_type == 'light'
  }

  print(f'Finding targets out of {len(values)} light-frame stacks')

  def comparison_fn(a, b):
    a_center, a_polygon = a
    b_center, b_polygon = b
    if np.dot(a_center, b_center) < new_context.cosine_similarity_threshold:
      return False

    iou = a_polygon.intersection(b_polygon).area() / a_polygon.union(
        b_polygon).area()
    return iou > 0.6

  groups = spatial_association.find_groups(values.keys(),
                                           values,
                                           comparison_fn,
                                           min_component_members=1)

  def new_target(stack_ids):
    center = stacks[stack_ids[0]].center
    name = position_string.canonical_target_string(center)

    footprint = next(
        SphericalPolygon.multi_intersection([
            SphericalPolygon(stacks[stack_id].footprint)
            for stack_id in stack_ids
        ]).points).tolist()
    projected_footprint = footprints.all_projections(
        SphericalPolygon(footprint))

    return Target(name=name,
                  center=center,
                  stack_ids=stack_ids,
                  footprint=footprint,
                  projected_footprint=projected_footprint)

  targets = [new_target(g) for g in groups]
  targets = {t.hash(): t for t in targets}

  print(f'Found {len(targets)} targets')

  # TODO: Computing the result and storing the result are two different
  # responsibilities. Perhaps separate patterns like this?
  new_context.targets = targets
