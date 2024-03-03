import ray
import datetime
import numpy as np
from spherical_geometry.polygon import SphericalPolygon
import astropy
import itertools

import stack
import clustering
from hashing import hash_string
import units
import footprints
from stack import Stack, Run, FrameHash


def _new_stack(name, light_frames, fits_headers, plate_solutions):
  flattened = list(itertools.chain.from_iterable([k for k in light_frames]))

  camera = next(iter(set(fits_headers[f]['INSTRUME'] for f in flattened)))
  filter = next(iter(set(fits_headers[f]['FILTER'] for f in flattened)))
  width_pixels = next(iter(set(fits_headers[f]['NAXIS1'] for f in flattened)))
  height_pixels = next(iter(set(fits_headers[f]['NAXIS2'] for f in flattened)))
  center = np.mean(
      [p.center for f, p in plate_solutions.items() if f in flattened], axis=0)
  image_scale = np.mean(
      [p.image_scale for f, p in plate_solutions.items() if f in flattened],
      axis=0)
  rotation_angle = np.mean(
      [p.rotation_angle for f, p in plate_solutions.items() if f in flattened],
      axis=0)
  footprint = next(
      SphericalPolygon.from_wcs(
          astropy.io.fits.Header(
              plate_solutions[flattened[0]].header)).points).tolist()
  projected_footprints = footprints.all_projections(SphericalPolygon(footprint))
  selection_criterion = ('StarCountFraction', 0.75)

  new_runs = [Run(frames=[FrameHash(h) for h in run]) for run in light_frames]

  return Stack(runs=new_runs,
               target=name,
               camera=camera,
               width_pixels=width_pixels,
               height_pixels=height_pixels,
               center=list(center),
               image_scale=image_scale,
               rotation_angle=rotation_angle,
               filter=filter,
               footprint=footprint,
               projected_footprints=projected_footprints,
               selection_criterion=selection_criterion)


@ray.remote
def identify_stacks(context):
  print('Identifying stacks...')

  fits_headers = ray.get(context.fits_headers)
  plate_solutions = ray.get(context.plate_solutions)
  spatial_groups = ray.get(context.spatial_groups)

  # TODO: Maybe implement a version of cluster_continuous() that repeatedly
  # subdivides by independent 1D variables.
  stacks = []
  for name, light_frames in spatial_groups:
    scales = [(f, plate_solutions[f].image_scale) for f in light_frames]
    for scale_cluster in clustering.cluster_continuous(scales, epsilon=0.2):
      scale_frames, _ = zip(*scale_cluster)
      cameras = [(f, (fits_headers[f]['INSTRUME'], fits_headers[f]['NAXIS1'],
                      fits_headers[f]['NAXIS2'], fits_headers[f]['XPIXSZ'],
                      fits_headers[f]['YPIXSZ'])) for f in scale_frames]
      for camera_cluster in clustering.cluster_discrete(cameras):
        camera_frames, cameras = zip(*camera_cluster)
        assert len(set(cameras)) == 1
        rotation_angles = [
            (f, plate_solutions[f].rotation_angle) for f in camera_frames
        ]
        for rotation_cluster in clustering.cluster_continuous(rotation_angles,
                                                              epsilon=5.0):
          rotation_frames, _ = zip(*rotation_cluster)
          filters = [(f, fits_headers[f]['FILTER']) for f in rotation_frames]
          for filter_cluster in clustering.cluster_discrete(filters):
            filter_frames, filters = zip(*filter_cluster)
            assert len(set(filters)) == 1
            date_obs = [(f, fits_headers[f]['DATE-OBS']) for f in filter_frames]
            light_frames = {}
            for run in clustering.cluster_continuous(
                date_obs, epsilon=datetime.timedelta(minutes=15)):
              light_frames_in_run, _ = zip(*run)

              # TODO: Put the minimum stack size in the UI.
              if len(light_frames_in_run) < 3:
                continue

              # TODO: This is a run of light frames with no dependencies,
              # but it is akwardly expressed here.
              light_frames[tuple(sorted(light_frames_in_run))] = set()

            flattened = list(
                itertools.chain.from_iterable([k for k in light_frames]))
            if len(flattened) >= 3:
              stacks.append(
                  _new_stack(name, light_frames, fits_headers, plate_solutions))

  print(f'Identified {len(stacks)} stacks')
  context.stacks = ray.put(stacks)
  return context
