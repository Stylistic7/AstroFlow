import ray
import numpy as np
from spherical_geometry.polygon import SphericalPolygon, SingleSphericalPolygon
import astropy
import datetime
import time
import objsize

from context import context
import clustering
from hashing import hash_string
import units
import geometry.footprints as footprints
from stacks import Stack, FrameHash, SimpleStack, CompositeStack
import units
from metadata.actor import get_metadata, cluster_by_header, sort_by_header
from profiling import track_performance


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
def _new_stacks(name, frames):
  metadata = [i for h, i in context.metadata.items() if h in frames]

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
      stack = SimpleStack(frames=run)
      stack.update_metadata()
      new_stacks[stack.content_hash] = stack

  if not new_stacks:
    return {}

  context.or_equals('stacks', new_stacks)

  # Add a toplevel stack, with the geometric information, that contains all
  # of the runs of light frames as its contents.
  stack = CompositeStack(stable_hash=stable_hash,
                         contents=list(new_stacks.keys()),
                         target=name,
                         center=list(center),
                         image_scale=image_scale,
                         rotation_angle=rotation_angle,
                         footprint=footprint,
                         projected_footprints=projected_footprints,
                         selection_criterion=selection_criterion)
  stack.update_metadata()
  new_stacks[stack.stable_hash] = stack
  context.or_equals('stacks', {stack.stable_hash: stack})

  return new_stacks


# TODO: Cache results for spatial groups. Astrophotographers rarely shoot more
# than a few different spatial positions in a night anyway. If the hashes of
# all the light frames are the same, then the resulting stacks are the same.
#
# TODO: The spatial_group should include a metadata_ref for all of the images
# in it. The call to get_metadata() should be removed.
@ray.remote
@track_performance
def identify_stacks(spatial_group):
  name, frames = spatial_group
  print(f'### identify_stacks running for '
        f'spatial group {name} of length {len(frames)}')

  if context.stacks == None:
    context.stacks = {}

  # TODO: Maybe implement a version of cluster_continuous() that repeatedly
  # subdivides by independent 1D variables.
  name, light_frames = spatial_group
  metadata = {h: i for h, i in context.metadata.items() if h in light_frames}

  def scale_clusters():
    scales = [(metadata[k].content_hash, metadata[k].plate_solution.image_scale)
              for k in metadata]
    for cluster in clustering.cluster_continuous(scales, epsilon=0.2):
      scale_frames, _ = zip(*cluster)
      yield scale_frames

  def camera_clusters(frames):
    cameras = [(metadata[k].content_hash,
                (metadata[k].header['INSTRUME'], metadata[k].header['NAXIS1'],
                 metadata[k].header['NAXIS2'], metadata[k].header['XPIXSZ'],
                 metadata[k].header['YPIXSZ'])) for k in frames]
    for cluster in clustering.cluster_discrete(cameras):
      frames, cameras = zip(*cluster)
      assert len(set(cameras)) == 1
      yield frames

  def rotation_clusters(frames):
    rotation_angles = [
        (metadata[k].content_hash,
         normalize_radians(metadata[k].plate_solution.rotation_angle))
        for k in frames
    ]
    for cluster in clustering.cluster_continuous(rotation_angles, epsilon=5.0):
      frames, _ = zip(*cluster)
      yield frames

  def filter_clusters(frames):
    filters = [
        (metadata[k].content_hash, metadata[k].header['FILTER']) for k in frames
    ]
    for cluster in clustering.cluster_discrete(filters):
      filter_frames, filters = zip(*cluster)
      assert len(set(filters)) == 1
      yield filter_frames

  start = time.time()
  new_stacks = {}
  for scale_frames in scale_clusters():
    for camera_frames in camera_clusters(scale_frames):
      for rotation_frames in rotation_clusters(camera_frames):
        for filter_frames in filter_clusters(rotation_frames):
          new_stacks |= _new_stacks(name, filter_frames)

  end = time.time()
  print(f'### identify_stacks took {end - start} seconds')
  return new_stacks
