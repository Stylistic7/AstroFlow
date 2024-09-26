import ray
import numpy as np
from spherical_geometry.polygon import SphericalPolygon, SingleSphericalPolygon  # type: ignore
import astropy  # type: ignore

from context import context
import clustering
from hashing import hash_string
import units
import geometry.footprints as footprints
from stacks import FrameHash, SimpleStack, CompositeStack, AstrometricAttributes, ImagingAttributes
import units
from profiling import track_performance
from metadata import MetadataDict
from stacks import StacksDict


def _normalize_radians(x: float) -> float:
  if x < 0.0:
    return x + np.pi
  else:
    return x


# TODO: Cache results for spatial groups. Astrophotographers rarely shoot more
# than a few different spatial positions in a night anyway. If the hashes of
# all the light frames are the same, then the resulting stacks are the same.
#
# TODO: The spatial_group should include a metadata_ref for all of the images
# in it. The call to get_metadata() should be removed.
#
# TODO: This is spilling huge amounts of memory, making it very slow.
@ray.remote
@track_performance
def identify_stacks(spatial_group):
  # TODO: Don't pass along the name here.
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
         _normalize_radians(metadata[k].plate_solution.rotation_angle))
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

  new_stacks = {}
  for scale_frames in scale_clusters():
    for camera_frames in camera_clusters(scale_frames):
      for rotation_frames in rotation_clusters(camera_frames):
        for filter_frames in filter_clusters(rotation_frames):
          new_stack = CompositeStack.from_frames(filter_frames, metadata)
          new_stacks[new_stack.stable_hash] = new_stack

  print(f'### identify_stacks done, writing {len(new_stacks)} '
        f'stacks into grouped_stacks')
  for stack in new_stacks.values():
    print(f'### identified stack {stack}')

  context.or_equals('grouped_stacks', new_stacks)
  return new_stacks
