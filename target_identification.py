from spherical_geometry.polygon import SphericalPolygon
import astropy.io.fits as pyfits
import ray

import file_io
import spatial_association
import position_string
from target import Target
import ray_diskcache
import footprints


@ray.remote
def group_stacks_into_targets(context):
  stacks = ray.get(context.stacks)
  fits_headers = ray.get(context.fits_headers)
  plate_solutions = ray.get(context.plate_solutions)

  def is_light_frame_stack(stack):
    for run in stack.runs:
      if not all(file_io.is_light_frame(fits_headers[f]) for f in run.frames):
        return False

    return True

  polygons = {
      id:
          SphericalPolygon.from_wcs(
              pyfits.Header(plate_solutions[stack.runs[0].frames[0]].header))
      for id, stack in stacks.items()
      if is_light_frame_stack(stack)
  }

  # TODO: Reject by cosine similarity first, for speed.
  def comparison_fn(a, b):
    iou = a.intersection(b).area() / a.union(b).area()
    return iou > 0.6

  groups = spatial_association.find_groups(polygons.keys(),
                                           polygons,
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

  # TODO: Computing the result and storing the result are two different
  # responsibilities. Perhaps separate patterns like this?
  ray_diskcache.set('targets', targets)

  context.targets = ray.put(targets)
  return context
