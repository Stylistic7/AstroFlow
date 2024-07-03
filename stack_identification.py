import ray
import numpy as np
from spherical_geometry.polygon import SphericalPolygon, SingleSphericalPolygon
import astropy
import datetime

from context import context
import clustering
from hashing import hash_string
import units
import footprints
from stack import Stack, FrameHash
import units
from image_metadata.actor import get_metadata, cluster_by_header, sort_by_header


def normalize_radians(x):
  if x < 0.0:
    return x + np.pi
  else:
    return x


# This hash defines the specific area of the sky this stack covers. It
# includes the center (RA, Dec), the rotation angle, the image scale, and
# the image width and height, in pixels. It is not defined for stacks that
# do not contain light frames.
def _compute_stable_hash(center, image_scale, rotation_angle, width_pixels,
                         height_pixels, filter) -> str:
  return hash_string(f'RA {units.ra_to_hms(center[0])}, '
                     f'Dec {units.dec_to_dms(center[1])}, '
                     f'Scale {image_scale:.2f}"/px, '
                     f'PA {rotation_angle:.2f}deg,'
                     f'Width {width_pixels}, '
                     f'Height {height_pixels}, '
                     f'Filter {filter}')


def _get_unique(L):
  s = set(L)
  assert len(s) == 1, f"list {L} has more than one unique member"
  return next(iter(s))


# TODO: The structure of light_frame_runs is completely nuts here.
def _new_stacks(name, light_frames):
  assert light_frames
  print(f'### _new_stacks, light_frames {light_frames}')

  width_pixels = _get_unique(
      get_metadata(light_frames, transform_fn=lambda i: i.header['NAXIS1']))
  height_pixels = _get_unique(
      get_metadata(light_frames, transform_fn=lambda i: i.header['NAXIS2']))

  # Sort the light frames by observation date. The oldest image will be used
  # to compute the hash of the whole stack.
  light_frames = sort_by_header(light_frames, 'DATE-OBS')
  print(f'### after sort, light_frames is {light_frames}')
  oldest_image = light_frames[0]

  oldest_image_metadata = get_metadata([oldest_image])[0]
  center = oldest_image_metadata.plate_solution.center
  assert list(center)

  image_scale = oldest_image_metadata.plate_solution.image_scale
  rotation_angle = normalize_radians(
      oldest_image_metadata.plate_solution.rotation_angle)

  filter = oldest_image_metadata.header['FILTER']

  stable_hash = _compute_stable_hash(center, image_scale, rotation_angle,
                                     width_pixels, height_pixels, filter)

  # TODO: Unify this WCS stuff and put it somewhere more general.
  # It's in target_identification and in mosaics/projection.
  footprint = SingleSphericalPolygon.from_wcs(
      astropy.io.fits.Header(oldest_image_metadata.plate_solution.header))
  if not footprint.is_clockwise():
    footprint = SingleSphericalPolygon(footprint.points[::-1])
  assert not SphericalPolygon.self_intersect(footprint.points)
  projected_footprints = footprints.all_projections(footprint)

  new_stacks = {}
  exposures = get_metadata(light_frames,
                           transform_fn=lambda i:
                           (i.content_hash, i.header['EXPOSURE']))
  for exposure_cluster in clustering.cluster_continuous(exposures,
                                                        epsilon=0.01):
    exposure_frames, exposures = zip(*exposure_cluster)
    assert np.all(np.isclose(exposures, exposures[0])
                 ), f'exposures {exposures} are clustered, but not all close'

    # TODO: The selection criterion needs to be preserved when stacks are
    # re-computed. Stacks with the same stable_hash can have their settings
    # passed on. There may be other settings like this in the future.
    selection_criterion = ('StarCountFraction', 0.75)
    runs = cluster_by_header(
        exposure_frames, 'DATE-OBS', lambda images: clustering.
        cluster_continuous(images, epsilon=context.max_run_gap))
    for run in runs:
      if len(run) < context.min_frames_in_run:
        continue
      run = [x[0] for x in run]
      print(f'### stack_identification._new_stacks making stack '
            f'with frames {run}')
      stack = Stack(frames=run)
      stack.update_metadata()
      print(f'### stack_identification._new_stacks made stack {stack}')
      new_stacks[stack.content_hash] = stack
      context.stacks |= {stack.content_hash: stack}

  # Add a toplevel stack, with the geometric information, that contains all
  # of the runs of light frames as its contents.
  stack = Stack(stable_hash=stable_hash,
                contents=list(new_stacks.keys()),
                target=name,
                center=list(center),
                image_scale=image_scale,
                rotation_angle=rotation_angle,
                footprint=footprint,
                projected_footprints=projected_footprints,
                selection_criterion=selection_criterion)
  stack.update_metadata()
  print(f'### stack_identification._new_stacks made toplevel stack {stack}')
  new_stacks[stack.stable_hash] = stack
  context.stacks |= {stack.stable_hash: stack}

  return new_stacks


@ray.remote
def identify_stacks():
  # TODO: Maybe just pass the spatial groups through the graph.
  spatial_groups = context.spatial_groups or []

  print(f'Identifying stacks from {len(spatial_groups)} spatial groups...')

  if context.stacks == None:
    context.stacks = {}

  # TODO: Maybe implement a version of cluster_continuous() that repeatedly
  # subdivides by independent 1D variables.
  #
  # TODO: A 6-deep nested for loop is pretty ugly. Can this be improved?
  debug_hash = None
  stacks = {}
  for name, light_frames in spatial_groups:
    scales = get_metadata(light_frames,
                          transform_fn=lambda i:
                          (i.content_hash, i.plate_solution.image_scale))
    print('### scales', scales)
    # scales = [(f, plate_solutions[f].image_scale) for f in light_frames]
    for scale_cluster in clustering.cluster_continuous(scales, epsilon=0.2):
      scale_frames, scales = zip(*scale_cluster)
      if debug_hash in scale_frames:
        print(f'Hash {debug_hash} found in cluster with scales {scales} '
              f'and frames {scale_frames}')
      cameras = get_metadata(
          scale_frames,
          transform_fn=lambda i:
          (i.content_hash, (i.header['INSTRUME'], i.header['NAXIS1'], i.header[
              'NAXIS2'], i.header['XPIXSZ'], i.header['YPIXSZ'])))
      # cameras = [(f, (fits_headers[f]['INSTRUME'], fits_headers[f]['NAXIS1'],
      #                 fits_headers[f]['NAXIS2'], fits_headers[f]['XPIXSZ'],
      #                 fits_headers[f]['YPIXSZ'])) for f in scale_frames]
      for camera_cluster in clustering.cluster_discrete(cameras):
        camera_frames, cameras = zip(*camera_cluster)
        if debug_hash in camera_frames:
          print(f'Hash {debug_hash} found in cluster with cameras {cameras} '
                f'and frames {camera_frames}')
        assert len(set(cameras)) == 1
        rotation_angles = get_metadata(
            predicate_fn=lambda i: i.content_hash in camera_frames,
            transform_fn=lambda i:
            (i.content_hash, normalize_radians(i.plate_solution.rotation_angle)
            ))
        # rotation_angles = [
        #     (f, normalize_radians(plate_solutions[f].rotation_angle))
        #     for f in camera_frames
        # ]
        for rotation_cluster in clustering.cluster_continuous(rotation_angles,
                                                              epsilon=5.0):
          rotation_frames, rotations = zip(*rotation_cluster)
          if debug_hash in rotation_frames:
            print(f'Hash {debug_hash} found in cluster with '
                  f'rotations {rotations} and frames {rotation_frames}')

          filters = get_metadata(
              predicate_fn=lambda i: i.content_hash in rotation_frames,
              transform_fn=lambda i: (i.content_hash, i.header['FILTER']))
          # filters = [(f, fits_headers[f]['FILTER']) for f in rotation_frames]
          for filter_cluster in clustering.cluster_discrete(filters):
            filter_frames, filters = zip(*filter_cluster)
            if debug_hash in filter_frames:
              print(f'Hash {debug_hash} found in cluster with '
                    f'filters {filters} and frames {filter_frames}')
            assert len(set(filters)) == 1
            stacks |= _new_stacks(name, filter_frames)

  print(f'Identified {len(stacks)} stacks')
  context.stacks = stacks
