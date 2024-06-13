import ray

from pathlib import Path
from enum import Enum
from dataclasses import dataclass

import stacking
from context import context
import pix
from cache import cache


@pix.PixInsight()
def _post_process_target(images):
  aligned = pix.StarAlignment(images)
  # TODO: Run ChannelCombination, ImageSolver, SPCC before doing
  # StarXTerminator.
  # TODO: ImageSolver must write the target's coordinates into the RGB image
  # so ImageSolver won't hang.
  return pix.StarXTerminator(aligned)


# TODO: Hash the script used to do the post-processing, and do not repeat
# the post-processing when a file with that hash already exists.
#
# TODO: The target and mosaic both need "stable_hashes" that are stable
# even when more images are added, and also "content_hashes" that indicate
# when images are added, or scripts are changed.
@ray.remote(resources={'PixInsight': 1}, scheduling_strategy="SPREAD")
def post_process_target(target_id):
  stacks = context.stacks
  targets = context.targets
  image_hashes = context.image_hashes
  fits_headers = context.fits_headers

  target = targets[target_id]
  print(f'Entering post_process_target for target {target}')

  filenames = []
  stacks = [stacks[s] for s in target.stack_ids]
  # TODO: Put the priority order in the UI.
  for filter in ['L', 'H', 'S', 'O', 'G', 'B', 'R']:
    matching_stack = next((s for s in stacks if s.filter == filter), None)
    if matching_stack:
      # TODO: Use the final file used by stack post-processing.
      masters = stacking.find_matching_masters(matching_stack, image_hashes,
                                               fits_headers)
      assert len(masters) == 1
      filenames.append(image_hashes[masters[0]])

  _post_process_target(filenames)


# # TODO: This is super-specific to my particular setup. If there's confusion about
# # which color channels to use, inform the user and stop.
# def order_stacks(stacks):
#   ordered_stacks = []
#   for filter in ['H', 'S', 'O', 'L', 'G', 'R', 'B']:
#     for rig in ['C11_F7', 'C11', 'RC71', 'TestRig']:
#       matching_stack = next(
#           (s for s in stacks if s.filter == filter and s.rig == rig), None)
#       if matching_stack:
#         ordered_stacks.append(matching_stack)
#         break

#   return ordered_stacks

# @pix.PixInsight
# def _post_process_target(target, stacks, dest_path):
#   # filenames = pix.FileList(s.post_processed for s in stacks)
#   filenames = [
#       get_prev_result_object('post_process_stack', s.id).filename
#       for s in stacks
#   ]
#   view_ids = pix.ViewIdList(s.filter for s in stacks)
#   print(f'### filenames {filenames}')
#   print(f'### view_ids {view_ids}')

#   dir_name = os.path.join(dest_path, 'Targets', target.name)
#   os.makedirs(dir_name, exist_ok=True)

#   view_ids = pix.OpenImages(filenames, view_ids)

#   if len(view_ids) > 1:
#     registered_view_ids = pix.StarAlignment(view_ids)
#     registered_filenames = [
#         os.path.join(dir_name, f'{s.filter}_registered.xisf') for s in stacks
#     ]
#     pix.SaveImages(registered_view_ids, registered_filenames)
#     # TODO: It would be super cool to automatically create these on saves.
#     saved_files = [
#         OutputFile(f, tags=dict(filter=s.filter, type='registered'))
#         for s, f in zip(stacks, registered_filenames)
#     ]
#   else:
#     registered_view_ids = view_ids
#     registered_filename = os.path.join(dir_name,
#                                        f'{stacks[0].filter}_registered.xisf')
#     shutil.copy(filenames[0], registered_filename)
#     saved_files = [
#         OutputFile(registered_filename,
#                    tags=dict(filter=[stacks[0].filter],
#                              image_type='registered'))
#     ]

#   # TODO: ViewIdList is leaking out of the API everywhere.
#   # TODO: The "vision" that abstracts views and filenames is really not
#   # working here at all.
#   if all(c in view_ids for c in 'RGB'):
#     registered_rgb_view_ids = pix.ViewIdList(
#         ['R_registered', 'G_registered', 'B_registered'])
#     rgb = pix.ChannelCombination(registered_rgb_view_ids, 'RGB')
#     # TODO: Automatically assign view ids to filenames.
#     rgb_filename = os.path.join(dir_name, 'RGB.xisf')
#     rgb = pix.SaveImages(rgb, pix.FileList([rgb_filename]))
#     # TODO: Include actual RGB stack ids.
#     saved_files.append(
#         OutputFile(rgb_filename,
#                    tags=dict(filter='RGB', image_type='registered')))
#     pix.CloseImages(registered_rgb_view_ids)

#     rgb = pix.ImageSolver(rgb)
#     rgb = pix.SpectrophotometricColorCalibration(rgb)
#     rgb_filename = os.path.join(dir_name, 'RGB_SPCC.xisf')
#     pix.SaveImages(rgb, pix.FileList([rgb_filename]))
#     saved_files.append(
#         OutputFile(rgb_filename,
#                    tags=dict(filter='RGB_SPCC', image_type='registered')))

#     registered_view_ids = pix.ViewIdList(
#         [r for r in registered_view_ids if r not in registered_rgb_view_ids])
#     registered_view_ids.append('RGB')

#   print('### registered_view_ids', registered_view_ids)
#   stars_view_ids = pix.StarXTerminator(registered_view_ids)

#   # These filenames are wrong, because I'm no longer using all of the stacks.
#   # The RGB images have been combined.
#   stars_filenames = [
#       os.path.join(dir_name, f'{v}.xisf') for v in stars_view_ids
#   ]
#   pix.SaveImages(stars_view_ids, stars_filenames)
#   # TODO: This is a chicken-and-egg problem. The PixInsight compute graph
#   # hasn't run yet, so the OutputFile can't get the image's hash yet.
#   # I need to wait until after the PixInsight function has completed to
#   # compute the image hash.
#   saved_files.extend([
#       OutputFile(f, tags=dict(filter=v, type='stars'))
#       for v, f in zip(stars_view_ids, stars_filenames)
#   ])

#   starless_filenames = [
#       os.path.join(dir_name, f'{v}_starless.xisf') for v in registered_view_ids
#   ]
#   pix.SaveImages(registered_view_ids, starless_filenames)
#   saved_files.extend([
#       OutputFile(f, tags=dict(filter=v, type='starless'))
#       for v, f in zip(registered_view_ids, starless_filenames)
#   ])

#   # TODO: Automatically keep track of saved filenames.
#   return saved_files

# def run(target, stacks, dest_path):
#   print(f'### Entering step_post_process_target.run() for target {target}')
#   post_processed = [
#       get_prev_result_object('post_process_stack', s).output_filename
#       for s in target.stack_ids
#   ]
#   print('### post_processed images are', post_processed)

#   if None in post_processed:
#     print(f'### Skipping target because some post-processed images are missing')
#     return (dg.Ran.RAN, (dg.Result.FAILURE,
#                          "One or more post-processed files are missing"))

#   # TODO: Add some kind of sentinel file to skip post-processing if it has
#   # already been done. (Maybe only use during development.)
#   if config.SUCCESS_IF_FILE_EXISTS:
#     dir_name = os.path.join(dest_path, 'Targets', target.name)
#     h_file = os.path.join(dir_name, 'H_registered.xisf')
#     if os.path.exists(h_file):
#       print(
#           '### step_post_process_target returning early because H file exists')
#       return (dg.Ran.RAN,
#               (dg.Result.SUCCESS,
#                [OutputFile(h_file, tags=dict(filter='H', type='registered'))]))

#   ordered_stacks = order_stacks([stacks[id] for id in target.stack_ids])
#   saved_files = _post_process_target(target, ordered_stacks, dest_path)

#   # TODO: Return FAILURE if the PixInsight task failed.
#   [f.hash_file(assert_exists=True) for f in saved_files]
#   return (dg.Ran.RAN, (dg.Result.SUCCESS, saved_files))
