import ray
import numpy as np
from spherical_geometry.polygon import SphericalPolygon
import astropy

from new_context import new_context
import ray_diskcache
import clustering
from hashing import hash_string
import units
import footprints
from stack import Stack, FrameHash
import units
from runs import get_runs, get_frames_for_run


def normalize_radians(x):
  if x < 0.0:
    return x + np.pi
  else:
    return x


# This hash defines the specific area of the sky this stack covers. It
# includes the center (RA, Dec), the rotation angle, the image scale, and
# the image width and height, in pixels. It is not defined for stacks that
# do not contain light frames.
def _compute_geometric_hash(center, image_scale, rotation_angle, width_pixels,
                            height_pixels, filter) -> str:
  return hash_string(f'RA {units.ra_to_hms(center[0])}, '
                     f'Dec {units.dec_to_dms(center[1])}, '
                     f'Scale {image_scale:.2f}"/px, '
                     f'PA {rotation_angle:.2f}deg,'
                     f'Width {width_pixels}, '
                     f'Height {height_pixels}, '
                     f'Filter {filter}')


# TODO: The structure of light_frame_runs is completely nuts here.
def _new_stacks(name, light_frames, fits_headers, plate_solutions):
  width_pixels = next(iter(set(
      fits_headers[f]['NAXIS1'] for f in light_frames)))
  height_pixels = next(
      iter(set(fits_headers[f]['NAXIS2'] for f in light_frames)))

  # Sort the light frames by observation date. The oldest image will be used
  # to compute the hash of the whole stack.
  light_frames = sorted(light_frames, key=lambda h: fits_headers[h]['DATE-OBS'])
  oldest_image = light_frames[0]

  center = plate_solutions[oldest_image].center
  assert list(center)

  image_scale = plate_solutions[oldest_image].image_scale
  rotation_angle = normalize_radians(
      plate_solutions[oldest_image].rotation_angle)

  filter = fits_headers[oldest_image]['FILTER']

  geometric_hash = _compute_geometric_hash(center, image_scale, rotation_angle,
                                           width_pixels, height_pixels, filter)

  footprint = next(
      SphericalPolygon.from_wcs(
          astropy.io.fits.Header(
              plate_solutions[light_frames[0]].header)).points).tolist()
  projected_footprints = footprints.all_projections(SphericalPolygon(footprint))

  new_stacks = {}
  exposures = [(f, fits_headers[f]['EXPOSURE']) for f in light_frames]
  for exposure_cluster in clustering.cluster_continuous(exposures,
                                                        epsilon=0.01):
    exposure_frames, exposures = zip(*exposure_cluster)
    assert np.all(np.isclose(exposures, exposures[0])
                 ), f'exposures {exposures} are clustered, but not all close'

    # TODO: The selection criterion needs to be preserved when stacks are
    # re-computed. Stacks with the same geometric_hash can have their settings
    # passed on. There may be other settings like this in the future.
    selection_criterion = ('StarCountFraction', 0.75)
    for run in get_runs(exposure_frames, fits_headers, new_context.max_run_gap,
                        new_context.min_frames_in_run):
      frames = get_frames_for_run(run, exposure_frames, fits_headers)
      assert len(frames)
      stack = Stack(frames=frames)
      stack.update_metadata(fits_headers)
      new_stacks[stack.content_hash] = stack

  # Add a toplevel stack, with the geometric information, that contains all
  # of the runs of light frames as its contents.
  stack = Stack(geometric_hash=geometric_hash,
                contents=list(new_stacks.keys()),
                target=name,
                center=list(center),
                image_scale=image_scale,
                rotation_angle=rotation_angle,
                footprint=footprint,
                projected_footprints=projected_footprints,
                selection_criterion=selection_criterion)
  stack.update_metadata(fits_headers, new_stacks)
  new_stacks[stack.geometric_hash] = stack

  return new_stacks


@ray.remote
def identify_stacks():
  fits_headers = new_context.fits_headers
  plate_solutions = new_context.plate_solutions
  spatial_groups = new_context.spatial_groups

  print(f'Identifying stacks from {len(spatial_groups)} spatial groups...')

  # TODO: Maybe implement a version of cluster_continuous() that repeatedly
  # subdivides by independent 1D variables.
  #
  # TODO: A 6-deep nested for loop is pretty ugly. Can this be improved?
  debug_hash = 'cbc3b0f593d8a78f'
  stacks = {}
  for name, light_frames in spatial_groups:
    scales = [(f, plate_solutions[f].image_scale) for f in light_frames]
    for scale_cluster in clustering.cluster_continuous(scales, epsilon=0.2):
      scale_frames, scales = zip(*scale_cluster)
      if debug_hash in scale_frames:
        print(f'Hash {debug_hash} found in cluster with scales {scales} '
              f'and frames {scale_frames}')
      cameras = [(f, (fits_headers[f]['INSTRUME'], fits_headers[f]['NAXIS1'],
                      fits_headers[f]['NAXIS2'], fits_headers[f]['XPIXSZ'],
                      fits_headers[f]['YPIXSZ'])) for f in scale_frames]
      for camera_cluster in clustering.cluster_discrete(cameras):
        camera_frames, cameras = zip(*camera_cluster)
        if debug_hash in camera_frames:
          print(f'Hash {debug_hash} found in cluster with cameras {cameras} '
                f'and frames {camera_frames}')
        assert len(set(cameras)) == 1
        rotation_angles = [
            (f, normalize_radians(plate_solutions[f].rotation_angle))
            for f in camera_frames
        ]
        for rotation_cluster in clustering.cluster_continuous(rotation_angles,
                                                              epsilon=5.0):
          rotation_frames, rotations = zip(*rotation_cluster)
          if debug_hash in rotation_frames:
            print(f'Hash {debug_hash} found in cluster with '
                  f'rotations {rotations} and frames {rotation_frames}')

          filters = [(f, fits_headers[f]['FILTER']) for f in rotation_frames]
          for filter_cluster in clustering.cluster_discrete(filters):
            filter_frames, filters = zip(*filter_cluster)
            if debug_hash in filter_frames:
              print(f'Hash {debug_hash} found in cluster with '
                    f'filters {filters} and frames {filter_frames}')
            assert len(set(filters)) == 1
            stacks |= _new_stacks(name, filter_frames, fits_headers,
                                  plate_solutions)

  print(f'Identified {len(stacks)} stacks')
  new_context.stacks = stacks
