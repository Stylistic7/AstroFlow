import tempfile
import os
import shutil
import subprocess
from pathlib import Path
import numpy as np
from xisf import XISF
from shapely.geometry import Polygon
from spherical_geometry.polygon import SphericalPolygon
import ray
from collections import defaultdict
import itertools

import config
import file_io
from stack import Stack, Status
import pix
from cache import cache
import step_flat_frame_association
from output_file import OutputFile
import spatial_association
import position_string
import ray_diskcache
import camera_configs
import runs


def _exposure_time_string(stack):
  exposure = stack.exposure_time_s
  hours = np.floor(exposure / 3600)
  minutes = np.floor(exposure % 3600 / 60)
  return f'{hours:02.0f}h{minutes:02.0f}m'


# TODO: Pull these functions into a new shared file.

# TODO: There are really three levels of confidence about using pre-existing
# artifacts:
#
# - Greatest confidence: The software has reached production quality, and
#   there's a reliable record of having run the task. The artifact also contains
#   unambiguous evidence of provenance (e.g. it contains a hashed stack id).
# - Middle confidence: The software is in development. There is no reliable
#   record of running the task, but there artifact contains unambiguous evidence
#   of provenance (e.g. it contains a hashed stack id).
# - Least confidence: The software is in development, or is run for the first
#   time on a new set of images. An artifact with the expected footprint exists,
#   but there is no evidence of its provenance.


def _scan_masters(path, image_hashes, plate_solutions):
  masters = {}
  for filename in Path(path).rglob("master*.xisf"):
    result = {}

    xisf = XISF(filename)
    file_meta = xisf.get_file_metadata()

    result['stack_id'] = file_meta.get('XISF:AstroFlow:StackId',
                                       {}).get('value', None)
    result['stack_content_hash'] = file_meta.get(
        'XISF:AstroFlow:StackContentHash', {}).get('value', None)

    hash = next((h for h, f in image_hashes.items() if f == filename), None)
    if hash and hash in plate_solutions:
      result['footprint'] = SphericalPolygon.from_wcs(
          plate_solutions[hash].header)
    else:
      result['footprint'] = None

    masters[filename] = result

  return masters


def _find_matching_master(stack, masters):
  for filename, metadata in masters.items():
    if metadata['stack_id'] == stack.id and metadata[
        'stack_content_hash'] == stack.content_hash:
      print(f'### Master {filename} matches {stack} by id and content hash')
      return filename


# def _master_with_id_or_footprint_exits(output_dir, stack, correct_filename,
#                                        image_hashes, plate_solutions):
#   print(f'### Entering _master_with_id_or_footprint_exits() for stack {stack}')
#   num_masters = 0
#   for filename in Path(output_dir).rglob("master*.xisf"):
#     num_masters += 1
#     xisf = XISF(filename)
#     ims_meta = xisf.get_images_metadata()

#     if ims_meta[0]['FITSKeywords']['FILTER'][0]['value'] != stack.filter:
#       continue

#     try:
#       xisf = XISF(filename)
#       file_meta = xisf.get_file_metadata()
#       stack_id_key = 'XISF:AstroFlow:StackId'
#       stack_content_hash_key = 'XISF:AstroFlow:StackContentHash'
#       if stack_id_key in file_meta and stack_content_hash_key in file_meta:
#         if file_meta[stack_id_key]['value'] == stack.id and file_meta[
#             stack_content_hash_key]['value'] == stack.content_hash:
#           print(f'### Found highly-likely master {filename} for '
#                 f'stack {stack} by XISF metadata')

#           if filename != correct_filename:
#             print(f'### Renaming master {filename} to {correct_filename}')
#             os.rename(filename, correct_filename)

#           return correct_filename
#     except Exception as e:
#       pass

#     try:

#       # TODO: Maybe use cosine similarity before IoU to speed this up.
#       if iou(SphericalPolygon(stack.footprint),
#              SphericalPolygon.from_wcs(plate_solutions[hash].header)) > 0.9:
#         print(f'### Found highly-likely master {filename} for '
#               f'stack {stack} by IoU')

#         if filename != correct_filename:
#           print(f'### Renaming master {filename} to {correct_filename}')
#           os.rename(filename, correct_filename)

#         return correct_filename
#     except Exception as e:
#       pass

#   print(f'### Inspected {num_masters} masters but none matched')


def _output_filename(stack, dest_path):
  output_dir = os.path.join(dest_path, 'Stacks', stack.target, 'masters')
  ra_dec_string = position_string.ra_dec_string(*stack.center)
  return os.path.join(
      output_dir, f'master_light_{stack.target}_{ra_dec_string}_'
      f'{stack.width_pixels}x{stack.height_pixels}_'
      f'{stack.image_scale:.02f}aspp_{stack.rotation_angle:.2f}deg_'
      f'{stack.filter}.xisf')


# @ray.remote
# def should_run(arg):
#   print('### Entering should_run()')
#   image_hashes, fits_headers, plate_solutions, dest_path, stack = arg

#   if stack.status != Status.READY:
#     return ResultDict(ran=True, run_successors=False)

#   dst_filename = _output_filename(stack, dest_path)
#   os.makedirs(os.path.dirname(dst_filename), exist_ok=True)
#   print(f'### step_stack.run dst_filename {dst_filename} exists '
#         f'{os.path.exists(dst_filename)}')

#   # The point of this is to short-circuit stacking when the cache empty, but
#   # the stacks have been done outside of AstroFlow, or the cache has been nuked.
#   # This generally happens when the user first installs AstroFlow.
#   if existing := _master_with_id_or_footprint_exits(stack, dst_filename,
#                                                     image_hashes,
#                                                     plate_solutions):
#     print(f'### Master already exists: {existing}')
#     return ResultDict(ran=True, run_successors=False)

#   return ResultDict(ran=True, run_successors=True)

# def resolve_files(filenames, dest_path, **kwargs):
#   smart_names = '_'.join([f'{k}_{v}' for k, v in kwargs.items()])
#   dst_filenames = []
#   for src_filename in filenames:
#     filename, ext = os.path.splitext(src_filename)
#     filename = os.path.basename(filename) + '_' + smart_names + ext
#     dst_filename = os.path.join(dest_path, filename)
#     print(f'src_filename {src_filename} symlinked as {dst_filename}')
#     dst_filenames.append(dst_filename)
#     # if not os.path.exists(dst_filename):
#     #   os.symlink(src_filename, dst_filename)

#   return dst_filenames

# def symlink_frames(filenames, dest_path, **kwargs):
#   suffix = '_' + '_'.join([f'{k}_{v}'
#                            for k, v in kwargs.items()]) if kwargs else ''
#   for filename in filenames:
#     path = Path(filename)
#     dest_filename = os.path.join(dest_path, f'{path.stem}{suffix}{path.suffix}')
#     print(f'### symlinking {filename} -> {dest_filename}')
#     os.symlink(filename, dest_filename)


def symlink_frames(filenames, dest_path):
  for filename in filenames:
    dest_filename = os.path.join(dest_path, os.path.basename(filename))
    print(f'### symlinking {filename} -> {dest_filename}')
    os.symlink(filename, dest_filename)


# TODO: Simplify this for the new dependency graph structure.
# `image_hashes` is only needed to find existing masters, and to get filenames
# for the light / dark / bias / flat frames in the stack. That could be done
# outside this function.
#
# This should be broken up into smaller pieces:
#
# should_run() checks for an existing master (local)
#
# resolve_files() makes lists of input files and their symlinked names,
#  which will include FLATVERS (local)
#
# symlink_and_run_wbpp() builds the temp directory with symlinks (local or remote)
#   pix.WeightedBatchPreprocessing() (local or remote)
#
# add_metadata() adds the stack id and content hash to the master (local or remote)
@ray.remote(resources={'PixInsight': 1}, scheduling_strategy="SPREAD")
def stack_subframes(context, stack):
  image_hashes = ray.get(context.image_hashes)
  fits_headers = ray.get(context.fits_headers)

  print(f'### Stacking subframes for {stack}')
  print(f'### Stack has {len(stack.light_frames)} light frames '
        f'{len(stack.dark_frames)} dark frames '
        f'{len(stack.bias_frames)} bias frames '
        f'{len(stack.flat_frames)} flat frames')

  with tempfile.TemporaryDirectory() as tmp_dir:
    for i, (light_run, flat_run) in enumerate(stack.flat_frame_associations):
      flatvers_dir = os.path.join(tmp_dir, f'FLATVERS_{i}')
      os.mkdir(flatvers_dir)

      light_frames = runs.get_frames_for_run(light_run, stack.light_frames,
                                             fits_headers)
      flat_frames = runs.get_frames_for_run(flat_run, stack.flat_frames,
                                            fits_headers)
      symlink_frames([image_hashes[h] for h in light_frames], flatvers_dir)
      symlink_frames([image_hashes[h] for h in flat_frames], flatvers_dir)

    symlink_frames([image_hashes[h] for h in stack.bias_frames], tmp_dir)
    symlink_frames([image_hashes[h] for h in stack.dark_frames], tmp_dir)

    pix.WeightedBatchPreprocessing(dir=tmp_dir, outputDirectory=tmp_dir)

    print('### WBPP master dir contains:')
    master_dir = os.path.join(tmp_dir, 'master')
    for f in os.listdir(master_dir):
      print(f'###   {f}')

    filenames = list(
        str(f) for f in Path(master_dir).rglob("*drizzle_2x_autocrop.xisf"))
    if not filenames:
      # TODO: Return failure, so the stack won't just be attempted again
      # on the next run. Distinguish the cases where the stack has never been
      # attempted from the cases where it just didn't succeed.
      print(f'### WBPP did not produce any master file.')
      return stack

    print('filenames', filenames)
    if len(filenames) != 1:
      print(f'### WARNING: stacking produced {len(filenames)} master files')

    # TODO: If there are exposures of different durations in the stack, WBPP
    # will produce multiple masters. Is this desirable? Should the user be
    # given multiple masters, so they can do HDR downstream? Or should I use
    # the WBPP exposure tolerance to make a single master?
    for filename in filenames:
      xisf = XISF(filename)
      image = xisf.read_image(0)
      file_meta = xisf.get_file_metadata()
      ims_meta = xisf.get_images_metadata()
      file_meta["XISF:AstroFlow:StackId"] = {
          "id": "XISF:AstroFlow:StackId",
          "type": "String",
          "value": stack.id,
      }
      file_meta["XISF:AstroFlow:StackContentHash"] = {
          "id": "XISF:AstroFlow:StackContentHash",
          "type": "String",
          "value": stack.content_hash,
      }

      dst_filename = _output_filename(stack, context.output_dir)
      XISF.write(dst_filename,
                 image,
                 creator_app="AstroFlow",
                 image_metadata=ims_meta[0],
                 xisf_metadata=file_meta)

      print('Stacking produced master', dst_filename)


@ray.remote(scheduling_strategy="SPREAD")
def scatter_stack_subframes(context):
  image_hashes = ray.get(context.image_hashes)
  plate_solutions = ray.get(context.plate_solutions)
  stacks = ray.get(context.stacks)
  print(f'### scatter_stack_subframes received {len(stacks)} stacks')

  masters = _scan_masters(context.output_dir, image_hashes, plate_solutions)
  print('### scanned masters:')
  for filename, metadata in masters.items():
    print(f'### filename {filename} metadata {metadata}')

  def artifact_exists(stack):
    matching_master = _find_matching_master(stack, masters)
    if not matching_master:
      return False

    print(f'### {stack} has existing artifact {matching_master}')

    correct_filename = _output_filename(stack, context.output_dir)
    if correct_filename != matching_master:
      os.makedirs(os.path.dirname(correct_filename), exist_ok=True)
      os.rename(matching_master, correct_filename)
      print(f'### Renamed master {matching_master} to {correct_filename}')

    return True

  stacks = [
      stack_subframes.bind(context, stack)
      for stack in stacks.values()
      if not artifact_exists(stack)
  ]
  print(f'### Scattering {len(stacks)} stacks that need stacking...')

  return ray.workflow.continuation(gather_stack_subframes.bind(context, stacks))


@ray.remote
def gather_stack_subframes(context, stacks):
  return context
