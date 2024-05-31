from pathlib import Path
from xisf import XISF
import astropy.io.fits as pyfits
import numpy as np
import ray

from context import context
from cache import cache
from stack import Stack
import stacking

# sys.path.append('/Users/nobody/Projects/Astro/GraXpert')
# from graxpert import background_extraction, background_grid_selection

import pix

# TODO: Pull these functions into a new shared file.
# def _post_processed_with_stack_id_exists(stack_id, filename):
#   if not os.path.exists(filename):
#     return False

#   try:
#     xisf = XISF(filename)
#     file_meta = xisf.get_file_metadata()
#     key = 'XISF:AstroFlow:StackId'
#     if key in file_meta:
#       print(f'### filename {filename} has XISF:AstroFlow:StackId',
#             file_meta[key]['value'])
#       if file_meta[key]['value'] == stack_id:
#         return True
#   except:
#     pass

#   print(f'### filename {filename} did not contain matching '
#         f'XISF:AstroFlow:StackId header')
#   return False

# def _output_filename(stack, dest_path):
#   output_dir = os.path.join(dest_path, 'Stacks', stack.target, 'masters')
#   return os.path.join(
#       output_dir,
#       f'post_processed_{stack.target}_{stack.width_pixels}x{stack.height_pixels}'
#       f'_{stack.image_scale:.02f}_{stack.filter}.xisf')


@pix.PixInsight()
def _post_process_stack(image):
  deconvolved = pix.BlurXTerminator(image)


@ray.remote(resources={'PixInsight': 1}, scheduling_strategy="SPREAD")
def post_process_stack(stack_id):
  stacks = context.stacks
  image_hashes = context.image_hashes
  fits_headers = context.fits_headers

  stack = stacks[stack_id]
  print(f'Entering post_process_stack for stack {stack}')

  masters = stacking.find_matching_masters(stack, image_hashes, fits_headers)
  if masters:
    filenames = [image_hashes[m] for m in masters]
    _post_process_stack(filenames)

  # master = get_prev_result_object('stack_subframes', stack.id)
  # print('master', master)
  # if not master:
  #   print('post_process_stack master is None')
  #   return (dg.Ran.RAN, (dg.Result.FAILURE, "Stack has no master"))

  # dst_filename = _output_filename(stack, dest_path)
  # print(f'### post_process_stack.run dst_filename {dst_filename} exists '
  #       f'{os.path.exists(dst_filename)}')

  # if config.TEST_MODE:
  #   return (dg.Ran.RAN, (dg.Result.SUCCESS,
  #                        dict(output_filename=dst_filename, snr=None)))

  # # The point of this is to short-circuit stacking when the cache is empty,
  # # but the stacks have already been done. This is useful during development.
  # # TODO: Make this a config option.
  # if _post_processed_with_stack_id_exists(stack.id, dst_filename):
  #   print(f'### Post-processed image (made by AstroFlow) already '
  #         f'exists: {dst_filename}')
  #   return (dg.Ran.DID_NOT_RUN, (dg.Result.SUCCESS,
  #                                dict(output_filename=dst_filename, snr=None)))

  # # TODO: Clean up the GraXpert integration.
  # # TODO: Move to the `logging` module.
  # xisf = XISF(master.filename)
  # image = xisf.read_image(0)
  # print(f'Before GraXpert, image has shape {image.shape} '
  #       f'type {image.dtype} min {np.min(image)} max {np.max(image)}')

  # if len(image.shape) == 2:
  #   image = np.expand_dims(image, 2)

  # background_points = background_grid_selection.background_grid_selection(
  #     image, num_pts_per_row=30, tol=1.5, sample_size=25)

  # # TODO: This will not work with RGB images.
  # def sample(x, y, _):
  #   return image[x - 12:x + 13, y - 12:y + 13, 0]

  # background_samples = [sample(*point) for point in background_points]
  # background_samples = [s for s in background_samples if s.shape == (25, 25)]
  # background_samples = np.stack(background_samples)
  # snr = np.mean(image) / np.std(background_samples)
  # print(f'### Image {master.filename} has SNR {snr}')

  # background = background_extraction.extract_background(
  #     image,
  #     np.array(background_points),
  #     interpolation_type='Kriging',
  #     smoothing=0.0,
  #     downscale_factor=4,
  #     sample_size=25,
  #     RBF_kernel='thin_plate',
  #     spline_order=5,
  #     corr_type='Subtraction',
  #     AI_dir='/Users/nobody/Projects/Astro/GraXpert/bg_model')
  # print(f'After GraXpert, image has shape {image.shape} '
  #       f'type {image.dtype} min {np.min(image)} max {np.max(image)}')
  # print(f'GraXpert background image has min {np.min(background)} '
  #       f'max {np.max(background)}')
  # # TODO: Use Exceptions that can be stored and shown to the user.
  # assert np.min(background) != 0.0 and np.max(background) != 0.0

  # filename, ext = os.path.splitext(master.filename)
  # graxpert_filename = f'graxpert_{filename}.xisf'
  # print(f'post_process_stack graxpert_filename {graxpert_filename}')

  # xisf = XISF(master.filename)
  # file_meta = xisf.get_file_metadata()
  # file_meta["XISF:AstroFlow:StackId"] = {
  #     "id": "XISF:AstroFlow:StackId",
  #     "type": "String",
  #     "value": stack.id,
  # }
  # ims_meta = xisf.get_images_metadata()
  # XISF.write(graxpert_filename,
  #            image,
  #            creator_app="AstroFlow",
  #            image_metadata=ims_meta[0],
  #            xisf_metadata=file_meta)

  # filename, ext = os.path.splitext(master.filename)
  # graxpert_background_filename = f'graxpert_background_{filename}.xisf'
  # XISF.write(graxpert_background_filename,
  #            background,
  #            creator_app="AstroFlow",
  #            image_metadata=ims_meta[0],
  #            xisf_metadata=file_meta)

  # _post_process_stack(stack, graxpert_filename, dst_filename)

  # # Add the XISF:AstroFlow:StackId header.
  # xisf = XISF(dst_filename)
  # file_meta = xisf.get_file_metadata()
  # file_meta["XISF:AstroFlow:StackId"] = {
  #     "id": "XISF:AstroFlow:StackId",
  #     "type": "String",
  #     "value": stack.id,
  # }
  # image = xisf.read_image(0)
  # ims_meta = xisf.get_images_metadata()
  # XISF.write(dst_filename,
  #            image,
  #            creator_app="AstroFlow",
  #            image_metadata=ims_meta[0],
  #            xisf_metadata=file_meta)

  # # TODO: Return FAILURE if PixInsight task failed for some reason.
  # return (dg.Ran.RAN, (dg.Result.SUCCESS,
  #                      dict(output_filename=dst_filename, snr=snr)))
