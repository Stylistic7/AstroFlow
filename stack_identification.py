import ray
import datetime
import numpy as np
from spherical_geometry.polygon import SphericalPolygon
import astropy

from stack import Stack
import clustering
from hashing import hash_string
import units
import footprints


def _new_stack(name, light_frames, fits_headers, plate_solutions):
  print('### Making stack with frames', light_frames)
  camera = next(iter(set(fits_headers[f]['INSTRUME'] for f in light_frames)))
  filter = next(iter(set(fits_headers[f]['FILTER'] for f in light_frames)))
  width_pixels = next(iter(set(
      fits_headers[f]['NAXIS1'] for f in light_frames)))
  height_pixels = next(
      iter(set(fits_headers[f]['NAXIS2'] for f in light_frames)))
  center = np.mean(
      [p.center for f, p in plate_solutions.items() if f in light_frames],
      axis=0)
  image_scale = np.mean(
      [p.image_scale for f, p in plate_solutions.items() if f in light_frames],
      axis=0)
  rotation_angle = np.mean([
      p.rotation_angle for f, p in plate_solutions.items() if f in light_frames
  ],
                           axis=0)
  id = hash_string(f'RA {units.ra_to_hms(center[0])}, '
                   f'Dec {units.dec_to_dms(center[1])}, '
                   f'Rotation {rotation_angle:.2f}, '
                   f'{width_pixels}x{height_pixels}, '
                   f'{image_scale:.2f}"/px, '
                   f'Filter {filter}')

  footprint = next(
      SphericalPolygon.from_wcs(
          astropy.io.fits.Header(
              plate_solutions[light_frames[0]].header)).points).tolist()
  projected_footprints = footprints.all_projections(SphericalPolygon(footprint))

  return Stack(id=id,
               content=light_frames,
               target=name,
               camera=camera,
               width_pixels=width_pixels,
               height_pixels=height_pixels,
               center=list(center),
               image_scale=image_scale,
               rotation_angle=rotation_angle,
               filter=filter,
               footprint=footprint,
               projected_footprints=projected_footprints)


@ray.remote
def identify_stacks_ray(context):
  print('Identifying stacks...')

  fits_headers = ray.get(context.fits_headers)
  plate_solutions = ray.get(context.plate_solutions)
  spatial_groups = ray.get(context.spatial_groups)

  # TODO: Maybe implement a version of cluster_continuous() that repeatedly
  # subdivides by independent 1D variables.
  stacks = []
  for name, light_frames in spatial_groups:
    scales = [(f, plate_solutions[f].image_scale) for f in light_frames]
    print('### scales', scales)
    for scale_cluster in clustering.cluster_continuous(scales, epsilon=0.2):
      scale_frames, _ = zip(*scale_cluster)
      cameras = [(f, fits_headers[f]['INSTRUME']) for f in scale_frames]
      print('### cameras', cameras)
      for camera_cluster in clustering.cluster_discrete(cameras):
        camera_frames, cameras = zip(*camera_cluster)
        assert len(set(cameras)) == 1
        rotation_angles = [
            (f, plate_solutions[f].rotation_angle) for f in camera_frames
        ]
        print('### rotation_angles', rotation_angles)
        for rotation_cluster in clustering.cluster_continuous(rotation_angles,
                                                              epsilon=5.0):
          rotation_frames, _ = zip(*rotation_cluster)
          filters = [(f, fits_headers[f]['FILTER']) for f in rotation_frames]
          print('### filters', filters)
          for filter_cluster in clustering.cluster_discrete(filters):
            filter_frames, filters = zip(*filter_cluster)
            assert len(set(filters)) == 1
            date_obs = [(f, fits_headers[f]['DATE-OBS']) for f in filter_frames]
            print('### date_obs', date_obs)
            for run in clustering.cluster_continuous(
                date_obs, epsilon=datetime.timedelta(minutes=15)):
              light_frames, _ = zip(*run)
              stacks.append(
                  _new_stack(name, light_frames, fits_headers, plate_solutions))

  print(f'Identified {len(stacks)} stacks')
  context.stacks = ray.put(stacks)
  return context
