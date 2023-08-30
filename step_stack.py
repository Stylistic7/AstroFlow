import tempfile
import os
import shutil
import subprocess
from pathlib import Path
import diskcache

import config
import file_io
from structs import Stack
import pixinsight_wbpp

cache = diskcache.Cache(os.path.join(os.getcwd(), '.cache'))


@cache.memoize(tag='step_stack')
def run(stack, dest_path):
  def symlink_frames(filenames, dest_path):
    for filename in filenames:
      dest_filename = os.path.join(dest_path, os.path.basename(filename))
      if not os.path.exists(dest_filename):
        os.symlink(filename, dest_filename)

  with tempfile.TemporaryDirectory() as tmp_dir:
    symlink_frames(stack.light_frames, tmp_dir)
    symlink_frames(stack.flat_frames, tmp_dir)
    symlink_frames(stack.bias_frames, tmp_dir)
    symlink_frames(stack.dark_frames, tmp_dir)

    src_master_dir = os.path.join(tmp_dir, 'master')

    if config.STACKING_TEST_MODE:
      # In test mode, don't actually stack anything. Just copy the first
      # light frame over.
      src_filename = stack.light_frames[0]
      dst_filename = os.path.join(dest_path, os.path.basename(src_filename))
      print(f'Stacking: test mode, returning light frame {src_filename}')
    else:
      pixinsight_wbpp.run(dir=tmp_dir, outputDirectory=tmp_dir)

      filenames = list(
          str(f)
          for f in Path(src_master_dir).rglob("*mono_drizzle_2x_autocrop.xisf"))
      if not filenames:
        # TODO: Return failure, so the stack won't just be attempted again
        # on the next run. Distinguish the cases where the stack has never been
        # attempted from the cases where it just didn't succeed.
        return None

      print('filenames', filenames)
      src_filename = filenames[0]

      while True:
        dst_filename = os.path.join(
            dest_path, stack.target, 'masters',
            f'master_light_{stack.target}_{stack.rig}_{stack.filter}.xisf')
        if not os.path.exists(dst_filename):
          break

    os.makedirs(os.path.join(dest_path, stack.target, 'masters'),
                exist_ok=True)
    shutil.copy(src_filename, dst_filename)
    print('Stacking produced master', dst_filename)
    return dst_filename