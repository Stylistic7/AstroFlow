import ray
import numpy as np
from spherical_geometry.polygon import SphericalPolygon, SingleSphericalPolygon
import astropy
import datetime
import time

from context import context
import clustering
from hashing import hash_string
import units
import geometry.footprints as footprints
from stacks import Stack, FrameHash
import units
from metadata.actor import get_metadata, cluster_by_header, sort_by_header


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


# TODO: The repeated calls to get_metadata() here, and in functions called
# by this one, is very slow.
#
# TODO: This could be parallelized.
def _new_stacks(name, metadata):
  start_time = time.time()

  print(f'### _new_stacks metadata type {type(metadata)} len {len(metadata)}')

  width_pixels = _get_unique(i.header['NAXIS1'] for i in metadata)
  height_pixels = _get_unique(i.header['NAXIS2'] for i in metadata)

  # Sort the light frames by observation date. The oldest image will be used
  # to compute the hash of the whole stack.
  sorted_metadata = sorted(metadata, key=lambda i: i.header['DATE-OBS'])
  oldest_image_metadata = sorted_metadata[0]

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
  exposures = [(i.content_hash, i.header['EXPOSURE'])
               for i in metadata
               if 'EXPOSURE' in i.header]
  for exposure_cluster in clustering.cluster_continuous(exposures,
                                                        epsilon=0.01):
    exposure_frames, exposures = zip(*exposure_cluster)
    print('### exposure_frames', exposure_frames)
    print('### exposures', exposures)
    assert np.all(np.isclose(exposures, exposures[0])
                 ), f'exposures {exposures} are clustered, but not all close'

    # TODO: The selection criterion needs to be preserved when stacks are
    # re-computed. Stacks with the same stable_hash can have their settings
    # passed on. There may be other settings like this in the future.
    selection_criterion = ('StarCountFraction', 0.75)
    runs = clustering.cluster_continuous([(i.content_hash, i.header['DATE-OBS'])
                                          for i in sorted_metadata
                                          if i.content_hash in exposure_frames],
                                         epsilon=context.max_run_gap)
    for run in runs:
      if len(run) < context.min_frames_in_run:
        continue
      run = [x[0] for x in run]
      stack = Stack(frames=run)
      stack.update_metadata(
          [i for i in metadata if i.content_hash in exposure_frames])
      print(f'### stack_identification._new_stacks made stack {stack}')
      new_stacks[stack.content_hash] = stack
      context.stacks |= {stack.content_hash: stack}

  if not new_stacks:
    return {}

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

  end_time = time.time()
  print(f'### _new_stacks elapsed time {end_time - start_time:.3f} s')

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
    metadata = {i.content_hash: i for i in get_metadata(light_frames)}
    scales = [(metadata[k].content_hash, metadata[k].plate_solution.image_scale)
              for k in metadata]
    for scale_cluster in clustering.cluster_continuous(scales, epsilon=0.2):
      scale_frames, scales = zip(*scale_cluster)
      if debug_hash in scale_frames:
        print(f'Hash {debug_hash} found in cluster with scales {scales} '
              f'and frames {scale_frames}')
      cameras = [(metadata[k].content_hash,
                  (metadata[k].header['INSTRUME'], metadata[k].header['NAXIS1'],
                   metadata[k].header['NAXIS2'], metadata[k].header['XPIXSZ'],
                   metadata[k].header['YPIXSZ'])) for k in scale_frames]
      for camera_cluster in clustering.cluster_discrete(cameras):
        camera_frames, cameras = zip(*camera_cluster)
        if debug_hash in camera_frames:
          print(f'Hash {debug_hash} found in cluster with cameras {cameras} '
                f'and frames {camera_frames}')
        assert len(set(cameras)) == 1
        rotation_angles = [
            (metadata[k].content_hash,
             normalize_radians(metadata[k].plate_solution.rotation_angle))
            for k in camera_frames
        ]
        for rotation_cluster in clustering.cluster_continuous(rotation_angles,
                                                              epsilon=5.0):
          rotation_frames, rotations = zip(*rotation_cluster)
          if debug_hash in rotation_frames:
            print(f'Hash {debug_hash} found in cluster with '
                  f'rotations {rotations} and frames {rotation_frames}')

          filters = [(metadata[k].content_hash, metadata[k].header['FILTER'])
                     for k in rotation_frames]
          for filter_cluster in clustering.cluster_discrete(filters):
            filter_frames, filters = zip(*filter_cluster)
            if debug_hash in filter_frames:
              print(f'Hash {debug_hash} found in cluster with '
                    f'filters {filters} and frames {filter_frames}')
            assert len(set(filters)) == 1
            new_stack_metadata = [
                v for k, v in metadata.items() if k in filter_frames
            ]
            stacks |= _new_stacks(name, new_stack_metadata)

  print(f'Identified {len(stacks)} stacks')
  context.stacks = stacks
