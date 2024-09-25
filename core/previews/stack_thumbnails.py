import numpy as np
import cv2
import os
import tempfile
import ray

from previews.operations import resize_array, array_to_png
from previews.flat_correction import create_flat_correction_thumbnails
from previews.calibration import calibrate_image
from context import context, WriteMode
from file_io.generic import read_image
# TODO: Unify quarantining
import quarantining
from stacks import SimpleStack, CompositeStack
from dependency_graph import Graph
from previews.operations import autostretch
from profiling import track_performance

# TODO: Split this file up into smaller, more specific pieces.

# Every stack is going to have multiple levels of previews. I should disentangle
# them. There will always be a subrame-based preview. The master-based preview
# will only be computed after stacking.
#
# The subframe-based preview requires calibration inside AstroFlow, but
# the master-based preview is just a thumbnail of the master image.
#
# There will also be post-processed previews. It would be good for the user
# to scroll through all the different types.
#
# The keys in the cache should be distinct:
#
# ('stack_subframe_thumbnail_image', 'abc123') = image
# ('stack_master_thumbnail_image', 'abc123') = image
# ('stack_postprocessed_thumbnail_image', 'abc123') = image
#
# Better:
#
# ('stack_thumbnail_image', 'subframe', 'abc123') = image
# ('stack_thumbnail_image', 'master', 'abc123') = image
# ('stack_thumbnail_image', 'post_processed', 'abc123') = image

MIN_STAR_FRACTION = 0.5


# TODO: This needs to use the Artifact workflow, so that it will automatically
# re-generate previews when stack contents change.
@ray.remote(resources={'fastest_storage': 0.001})
def scatter_stack_thumbnails(mode):
  stacks = context.stacks
  cache_keys = context.get_cache_keys()

  if mode == 'subframe':
    subframe_images = [(create_flat_correction_thumbnails, id)
                       for id, s in stacks.items()
                       if type(s) is SimpleStack and s.image_type == 'light']
    subframe_videos = [
        (create_stack_subframe_thumbnails, id)
        for id, s in stacks.items()
        if type(s) is CompositeStack and s.image_type == 'light' and (
            ('stack_subframe_thumbnail_image', id) not in cache_keys or
            ('stack_subframe_thumbnail_video', id) not in cache_keys)
    ]
    tasks = subframe_images + subframe_videos
  elif mode == 'master':
    tasks = [(create_stack_master_thumbnail, id)
             for id, s in stacks.items()
             if type(s) is CompositeStack and s.image_type == 'light' and
             ('stack_master_thumbnail_image', id) not in cache_keys]

  return Graph({gather: tasks})


# TODO: The thumbnail images and videos need to use the artifact workflow, so
# they will be regenerated after quarantining, etc.
@ray.remote
@track_performance
def create_stack_subframe_thumbnails(stack_id):
  stacks = context.stacks
  metadata = context.metadata

  stack = stacks[stack_id]
  print(f'Entering create_stack_subframe_thumbnails() for stack {stack}')

  # Don't make previews for calibration stacks.
  assert type(stack) is not SimpleStack

  # settings_key = ('stack_settings', stack_id)
  # settings = context[settings_key] or {}
  # FIXME the dependencies have to be taken from the substacks, e.g. the
  # light runs. But the sorting by number of stars makes it hard to look up
  # the substack it belonged to.
  # dependencies = stack.dependencies | settings

  width = context.subframe_thumbnail_width

  # Cache the calibration images so they're only read once.
  first_subframes = {}

  def get_first_subframe(stack_id):
    if not stack_id:
      return None
    elif stack_id in first_subframes:
      return first_subframes[stack_id]
    else:
      frame = metadata[stacks[stack_id].frames[0]]
      image = resize_array(read_image(frame.filename), width).astype(np.float32)
      first_subframes[stack_id] = (frame, image)
      return first_subframes[stack_id]

  # Now sort all the light frames (no matter which run they're from) by the
  # number of stars. The better images should be in front.
  light_frames_by_stars = quarantining.get_light_frames_sorted_by_stars(
      stack.all_frames(), metadata)
  best_light_frame = light_frames_by_stars[0]
  image = resize_array(read_image(best_light_frame.filename), width)
  height, width = image.shape

  height = int(((height + 8 - 1) / 8) * 8)
  print(f'Creating video of width {width}, height {height} '
        f'with {len(light_frames_by_stars)} frames')

  with tempfile.TemporaryDirectory() as tmp_dir:
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    video_filename = os.path.join(tmp_dir, 'output.webm')
    out = cv2.VideoWriter(video_filename, fourcc, 10.0, (width, height), False)
    for i, light_frame in enumerate(light_frames_by_stars):
      rotate = light_frame.plate_solution.rotation_angle < 0.0

      light_image = resize_array(read_image(light_frame.filename),
                                 width).astype(np.float32)

      # TODO: This feels a little hacky.
      # FIXME and it isn't working
      dependencies = {}
      if substack_id := next(
          iter(s for s in stack.contents
               if light_frame.content_hash in stacks[s].frames), None):
        dependencies = stacks[substack_id].dependencies

      bias_frame = bias_image = None
      if 'bias' in dependencies:
        bias_frame, bias_image = get_first_subframe(dependencies['bias'])

      dark_frame = dark_image = None
      if 'dark' in dependencies:
        dark_frame, dark_image = get_first_subframe(dependencies['dark'])

      flat_frame = flat_image = None
      if 'flat' in dependencies:
        flat_frame, flat_image = get_first_subframe(dependencies['flat'])

      if bias_frame is None or dark_frame is None or flat_frame is None:
        print(f'### create_stack_subframe_thumbnails light frame '
              f'{light_frame.content_hash} from stack {stacks[substack_id]} '
              f'is missing some calibration images: {dependencies}')

      im = calibrate_image(light_frame, light_image, bias_frame, bias_image,
                           dark_frame, dark_image, flat_frame, flat_image,
                           rotate)
      im = autostretch(np.clip(im, 0.0, 1.0))

      if i == 0:
        key = ('stack_subframe_thumbnail_image', stack_id)
        context.set(key, array_to_png(im), write_mode=WriteMode.DISK_ONLY)
        print(f'### Wrote key {key}')

      im = (im * 255.0).astype(np.uint8)
      frame = np.zeros((height, width), dtype=np.uint8)
      frame[0:im.shape[0], 0:im.shape[1]] = im

      out.write(frame)

    out.release()
    thumbnail_video = open(video_filename, 'rb').read()

    key = ('stack_subframe_thumbnail_video', stack_id)
    context.set(key, thumbnail_video, write_mode=WriteMode.DISK_ONLY)
    print(f'### Wrote key {key}')


@ray.remote
def gather():
  pass
