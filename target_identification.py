from spherical_geometry.polygon import SphericalPolygon
import astropy.io.fits as pyfits
import ray

import spatial_association
import position_string
from target import Target
import ray_diskcache


@ray.remote
def group_stacks_into_targets(context):
  stacks = ray.get(context.stacks)
  plate_solutions = ray.get(context.plate_solutions)

  polygons = {
      s.id:
          SphericalPolygon.from_wcs(
              pyfits.Header(plate_solutions[s.light_frames[0]].header))
      for s in stacks.values()
  }

  # TODO: Stop using IoU, and use cosine similarity instead.
  def comparison_fn(a, b):
    iou = a.intersection(b).area() / a.union(b).area()
    return iou > 0.6

  groups = spatial_association.find_groups(polygons.keys(),
                                           polygons,
                                           comparison_fn,
                                           min_component_members=1)
  print('### Found groups', groups)

  def new_target(stack_ids):
    center = stacks[stack_ids[0]].center
    print('### new_target center', center)
    name = position_string.canonical_target_string(center)
    target = Target(name=name, stack_ids=stack_ids)
    target.store_metadata(stacks)
    return target

  targets = [new_target(g) for g in groups]
  targets = {t.id: t for t in targets}

  # TODO: Computing the result and storing the result are two different
  # responsibilities. Perhaps separate patterns like this?
  ray_diskcache.set('targets', targets)

  context.targets = ray.put(targets)
  return context
