import tempfile
import os
from pathlib import Path
import numpy as np
from xisf import XISF
from spherical_geometry.polygon import SphericalPolygon
import ray

import pix
import position_string
import runs
import hashing

_STACK_ID_KEY = 'XISF:AstroFlow:StackId'
_CONTENT_HASH_KEY = 'XISF:AstroFlow:StackContentHash'


def _exposure_time_string(stack):
  exposure = stack.exposure_time_s
  hours = np.floor(exposure / 3600)
  minutes = np.floor(exposure % 3600 / 60)
  return f'{hours:02.0f}h{minutes:02.0f}m'


# TODO: Remove this.
def _scan_masters(path, image_hashes, plate_solutions):
  masters = {}
  for filename in Path(path).rglob("master*.xisf"):
    result = {}

    xisf = XISF(filename)
    file_meta = xisf.get_file_metadata()

    result['stack_id'] = file_meta.get(_STACK_ID_KEY, {}).get('value', None)
    result['stack_content_hash'] = file_meta.get(_CONTENT_HASH_KEY,
                                                 {}).get('value', None)

    hash = next((h for h, f in image_hashes.items() if f == filename), None)
    if hash and hash in plate_solutions:
      result['footprint'] = SphericalPolygon.from_wcs(
          plate_solutions[hash].header)
    else:
      result['footprint'] = None

    masters[filename] = result

  return masters


def find_matching_masters(fits_headers, content_hash):
  return [
      h for h, header in fits_headers.items() if _CONTENT_HASH_KEY in header and
      header[_CONTENT_HASH_KEY] == content_hash
  ]


def _output_filename(stack, dest_path):
  output_dir = os.path.join(dest_path, 'Stacks', stack.target, 'masters')
  ra_dec_string = position_string.ra_dec_string(*stack.center)
  return os.path.join(
      output_dir, f'master_light_{stack.target}_{ra_dec_string}_'
      f'{stack.width_pixels}x{stack.height_pixels}_'
      f'{stack.image_scale:.02f}aspp_{stack.rotation_angle:.2f}deg_'
      f'{stack.filter}.xisf')


def _symlink_frames(filenames, dest_path):
  for filename in filenames:
    dest_filename = os.path.join(dest_path, os.path.basename(filename))
    print(f'### symlinking {filename} -> {dest_filename}')
    os.symlink(filename, dest_filename)


def _symlink_stack_frames(stack, image_hashes, fits_headers, dest_path):
  print(f'### Stack has {len(stack.light_frames)} light frames '
        f'{len(stack.dark_frames)} dark frames '
        f'{len(stack.bias_frames)} bias frames '
        f'{len(stack.flat_frames)} flat frames')

  master_biases = find_matching_masters(fits_headers,
                                        hashing.hash_list(stack.bias_frames))
  master_darks = find_matching_masters(fits_headers,
                                       hashing.hash_list(stack.dark_frames))

  assert len(master_biases) < 2
  assert len(master_darks) < 2

  if stack.flat_frame_associations:
    for i, (light_run, flat_run) in enumerate(stack.flat_frame_associations):
      flatvers_dir = os.path.join(dest_path, f'FLATVERS_{i}')
      os.mkdir(flatvers_dir)

      light_frames = runs.get_frames_for_run(light_run, stack.light_frames,
                                             fits_headers)
      flat_frames = runs.get_frames_for_run(flat_run, stack.flat_frames,
                                            fits_headers)
      master_flats = find_matching_masters(fits_headers,
                                           hashing.hash_list(flat_frames))

      _symlink_frames([image_hashes[h] for h in light_frames], flatvers_dir)
      _symlink_frames(master_flats, flatvers_dir)
  else:
    _symlink_frames([image_hashes[h] for h in stack.light_frames], dest_path)
    master_flats = find_matching_masters(fits_headers,
                                         hashing.hash_list(stack.flat_frames))
    _symlink_frames(master_flats, dest_path)

  _symlink_frames(master_biases, dest_path)
  _symlink_frames(master_darks, dest_path)


def _add_xisf_metadata(filename, dest_filename, metadata):
  xisf = XISF(filename)
  image = xisf.read_image(0)
  file_meta = xisf.get_file_metadata()
  ims_meta = xisf.get_images_metadata()

  for k, v in metadata.items():
    file_meta[k] = {
        "id": k,
        "type": "String",
        "value": v,
    }

  XISF.write(dest_filename,
             image,
             creator_app="AstroFlow",
             image_metadata=ims_meta[0],
             xisf_metadata=file_meta)


@ray.remote(resources={'PixInsight': 1}, scheduling_strategy="SPREAD")
def stack_files(context, hashes, dest_path, metadata=None):
  print(f'### Entering stack_files for {len(hashes)} images')

  image_hashes = ray.get(context.image_hashes)
  filenames = [image_hashes[h] for h in hashes]

  if not metadata:
    metadata = {}

  if _CONTENT_HASH_KEY not in metadata:
    metadata[_CONTENT_HASH_KEY] = hashing.hash_list(hashes)

  with tempfile.TemporaryDirectory() as tmp_dir:
    _symlink_frames(filenames, tmp_dir)
    pix.WeightedBatchPreprocessing(dir=tmp_dir, outputDirectory=tmp_dir)

    master_dir = os.path.join(tmp_dir, 'master')
    if not os.path.exists(master_dir):
      print(f'### WBPP did not produce an output master dir.')
      return None

    print('### WBPP master dir contains:')
    wbpp_filenames = os.listdir(master_dir)
    for f in wbpp_filenames:
      print(f'###   {f}')

    output_filenames = []
    for filename in wbpp_filenames:
      filename = os.path.join(master_dir, wbpp_filenames[0])
      dest_filename = os.path.join(dest_path, os.path.basename(filename))
      _add_xisf_metadata(filename, dest_filename, metadata)
      output_filenames.append(dest_filename)

    return output_filenames


@ray.remote(resources={'PixInsight': 1}, scheduling_strategy="SPREAD")
def stack_stack(context, stack):
  print(f'### Stacking subframes for {stack}')

  image_hashes = ray.get(context.image_hashes)
  fits_headers = ray.get(context.fits_headers)

  with tempfile.TemporaryDirectory() as tmp_dir:
    _symlink_stack_frames(stack, image_hashes, fits_headers, tmp_dir)

    pix.WeightedBatchPreprocessing(dir=tmp_dir, outputDirectory=tmp_dir)

    master_dir = os.path.join(tmp_dir, 'master')
    if not os.path.exists(master_dir):
      print(f'### WBPP did not produce an output master dir.')
      return stack

    print('### WBPP master dir contains:')
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
      dest_filename = _output_filename(stack, context.output_dir)
      metadata = {
          _STACK_ID_KEY: stack.id,
          _CONTENT_HASH_KEY: stack.content_hash,
      }
      os.makedirs(os.path.dirname(dest_filename), exist_ok=True)
      _add_xisf_metadata(filename, dest_filename, metadata)
      print('Stacking produced master', dest_filename)


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
    if str(matching_master) != str(correct_filename):
      os.makedirs(os.path.dirname(correct_filename), exist_ok=True)
      os.rename(matching_master, correct_filename)
      print(f'### Renamed master {matching_master} to {correct_filename}')

    return True

  stacks = [
      stack_stack.remote(context, stack)
      for stack in stacks.values()
      if not artifact_exists(stack)
  ]
  print(f'### Scattering {len(stacks)} stacks that need stacking...')

  return gather_stack_subframes.remote(context, stacks)


@ray.remote
def gather_stack_subframes(context, stacks):
  return context
