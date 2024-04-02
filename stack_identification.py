import ray
import datetime
import numpy as np
from spherical_geometry.polygon import SphericalPolygon
import astropy
import itertools

import stack
import clustering
from hashing import hash_string, hash_list
import units
import footprints
from stack import Stack, Run, FrameHash
import units
import subframe_selection
from runs import get_runs

from constants import kRunMaxGap, kMinFramesInStack


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
                            height_pixels) -> str:
  return hash_string(f'RA {units.ra_to_hms(center[0])}, '
                     f'Dec {units.dec_to_dms(center[1])}, '
                     f'Scale {image_scale:.2f}"/px, '
                     f'PA {rotation_angle:.2f}deg,'
                     f'Width {width_pixels}, Height {height_pixels}')


# This hash includes runs of frames, e.g. light frames, plus all other stacks
# that this stack depends on, e.g. stacks of bias and dark frames.
def _compute_content_hash(runs) -> str:
  all_frames = []
  for r in runs:
    all_frames.extend(r.frames)
    all_frames.extend(r.stack_dependencies)

  return hash_list(sorted(all_frames))


def _new_stack(name, light_frames, fits_headers, plate_solutions):
  print(f'Entering _new_stack, light_frames are:')
  for run in light_frames:
    print(f'run {run}')
    assert len(get_runs(run, fits_headers)) == 1

  flattened = list(itertools.chain.from_iterable([k for k in light_frames]))

  camera = next(iter(set(fits_headers[f]['INSTRUME'] for f in flattened)))
  filter = next(iter(set(fits_headers[f]['FILTER'] for f in flattened)))
  width_pixels = next(iter(set(fits_headers[f]['NAXIS1'] for f in flattened)))
  height_pixels = next(iter(set(fits_headers[f]['NAXIS2'] for f in flattened)))

  # Sort the light frames by observation date. The oldest image will be used
  # to compute the hash of the whole stack.
  flattened = sorted(flattened, key=lambda h: fits_headers[h]['DATE-OBS'])
  oldest_image = flattened[0]

  center = plate_solutions[oldest_image].center
  image_scale = plate_solutions[oldest_image].image_scale
  rotation_angle = normalize_radians(
      plate_solutions[oldest_image].rotation_angle)

  geometric_hash = _compute_geometric_hash(center, image_scale, rotation_angle,
                                           width_pixels, height_pixels)

  footprint = next(
      SphericalPolygon.from_wcs(
          astropy.io.fits.Header(
              plate_solutions[flattened[0]].header)).points).tolist()
  projected_footprints = footprints.all_projections(SphericalPolygon(footprint))

  # TODO: The selection criterion needs to be preserved when stacks are
  # re-computed. Stacks with the same geometric_hash can have their settings
  # passed on. There may be other settings like this in the future.
  selection_criterion = ('StarCountFraction', 0.75)

  runs = [Run(frames=[FrameHash(h) for h in run]) for run in light_frames]

  for run in runs:
    assert run.frames, f'run.frames {run.frames} was Falsey'
    print(f'run {run}')
    assert len(get_runs(run.frames, fits_headers)) == 1

  num_light_frames_before_selection = len(
      [f for run in runs for f in run.frames])
  print(f'Before down-selection, {num_light_frames_before_selection} frames '
        f'in {len(runs)} runs')
  criterion = subframe_selection.decode_selection_criterion(selection_criterion)
  runs = criterion.select_subframes(runs, fits_headers, plate_solutions)
  num_light_frames_after_selection = len(
      [f for run in runs for f in run.frames])
  print(f'After down-selection, {num_light_frames_after_selection} frames '
        f'in {len(runs)} runs')

  for run in runs:
    assert run.frames, f'run.frames {run.frames} was Falsey'
    print(f'run {run}')
    assert len(get_runs(run.frames, fits_headers)) == 1

  # TODO: Put this number in the UI somewhere.
  if num_light_frames_after_selection < kMinFramesInStack:
    return None

  content_hash = _compute_content_hash(runs)
  assert content_hash

  return Stack(geometric_hash=geometric_hash,
               content_hash=content_hash,
               runs=runs,
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
            (f, normalize_radians(plate_solutions[f].rotation_angle))
            for f in camera_frames
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
                date_obs, epsilon=datetime.timedelta(minutes=kRunMaxGap)):
              light_frames_in_run, _ = zip(*run)

              # TODO: This is a run of light frames with no dependencies,
              # but it is akwardly expressed here.
              light_frames[tuple(sorted(light_frames_in_run))] = set()

            new_stack = _new_stack(name, light_frames, fits_headers,
                                   plate_solutions)
            if new_stack:
              stacks.append(new_stack)

  print(f'Identified {len(stacks)} stacks')
  context.stacks = ray.put(stacks)
  return context, None
