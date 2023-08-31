import os
import diskcache

import spatial_association

cache = diskcache.Cache(os.path.join(os.getcwd(), '.cache'))


def run(plate_solutions):
  # TODO: This should only group the light frames actually selected, even
  # if there are more plate solutions in the artifact database.
  #
  # TODO: All of the spatial positions should be printed out, along with their
  # common directory. I need to make sure the tolerances (0.1 degree) are
  # actually reasonable, and won't ever get confused by dither.
  spatial_groups = spatial_association.find_components(
      list(sorted(plate_solutions.keys())))

  for target, g in spatial_groups:
    print(f"Spatial group '{target}' has {len(g)} light frames")

  return spatial_groups