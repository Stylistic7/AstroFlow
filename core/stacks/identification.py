import numpy as np
from beartype import beartype

from typing import Iterator
from context import context
import clustering
from stacks.composite_stack import CompositeStack, CompositeStacksDict
from metadata.classes import FrameHash, MetadataDict
from decorators import cached


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
@cached.remote
def identify_stacks(light_frames: list[FrameHash]) -> CompositeStacksDict:
  print(f'### identify_stacks running for spatial group of '
        f'length {len(light_frames)}')

  # TODO: Maybe implement a version of cluster_continuous() that repeatedly
  # subdivides by independent 1D variables.
  metadata = MetadataDict({
      h: i for h, i in context.metadata.items() if h in light_frames
  })

  @beartype
  def scale_clusters() -> Iterator[tuple[FrameHash, ...]]:
    scales = [(metadata[k].content_hash, metadata[k].plate_solution.image_scale)
              for k in metadata]
    for cluster in clustering.cluster_continuous(scales, epsilon=0.2):
      scale_frames, _ = zip(*cluster)
      yield scale_frames

  @beartype
  def camera_clusters(
      frames: tuple[FrameHash, ...]) -> Iterator[tuple[FrameHash, ...]]:
    cameras = [(metadata[k].content_hash,
                (metadata[k].header['INSTRUME'], metadata[k].header['NAXIS1'],
                 metadata[k].header['NAXIS2'], metadata[k].header['XPIXSZ'],
                 metadata[k].header['YPIXSZ'])) for k in frames]
    for cluster in clustering.cluster_discrete(cameras):
      frames, cameras = zip(*cluster)
      assert len(set(cameras)) == 1
      yield frames

  @beartype
  def rotation_clusters(
      frames: tuple[FrameHash, ...]) -> Iterator[tuple[FrameHash, ...]]:
    rotation_angles = [
        (metadata[k].content_hash,
         _normalize_radians(metadata[k].plate_solution.rotation_angle))
        for k in frames
    ]
    for cluster in clustering.cluster_continuous(rotation_angles, epsilon=5.0):
      frames, _ = zip(*cluster)
      yield frames

  @beartype
  def filter_clusters(
      frames: tuple[FrameHash, ...]) -> Iterator[tuple[FrameHash, ...]]:
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
          if filter_frames:
            new_stack = CompositeStack.from_frames(filter_frames,
                                                   metadata=metadata)
            if new_stack:
              new_stacks[new_stack.stable_hash] = new_stack

  print(f'### identify_stacks done, returning {len(new_stacks)} stacks')
  return new_stacks
