import tempfile
import os
import shutil
import subprocess
from pathlib import Path

import config
import file_io
import flat_frame_association
import dark_and_bias_association


def stack(position_string, rig, filter, light_frames, flat_frames, version):
  def symlink_frames(filenames, dest_path):
    for filename in filenames:
      dest_filename = os.path.join(dest_path, os.path.basename(filename))
      if not os.path.exists(dest_filename):
        os.symlink(filename, dest_filename)

  flat_frame_association.associate_flat_frames(light_frames, flat_frames)

  bias_frames = []
  dark_frames = []
  camera_configs = dark_and_bias_association.get_unique_camera_configs(
      light_frames)
  for camera_config in camera_configs:
    camera, temperature, exposure = camera_config
    bias_frames += dark_and_bias_association.find_bias_frames(
        camera, temperature)
    dark_frames += dark_and_bias_association.find_dark_frames(
        camera, temperature, exposure)

  with tempfile.TemporaryDirectory() as tmp_dir:
    symlink_frames(light_frames, tmp_dir)
    symlink_frames(flat_frames, tmp_dir)
    symlink_frames(bias_frames, tmp_dir)
    symlink_frames(dark_frames, tmp_dir)

    src_master_dir = os.path.join(tmp_dir, 'master')
    dst_master_dir = os.path.commonpath(light_frames)
    os.makedirs(dst_master_dir, exist_ok=True)

    print('stacking: src_master_dir', src_master_dir)
    print('stacking: dst_master_dir', dst_master_dir)

    if config.STACKING_TEST_MODE:
      filter = file_io.read_fits(light_frames[0])['FILTER']
      master_filename = f'masterLight_BIN-1_6248x4176_EXPOSURE-60.00s_FILTER-{filter}_mono_drizzle_2x_autocrop.xisf'

      os.makedirs(src_master_dir, exist_ok=True)
      open(os.path.join(src_master_dir, master_filename), 'w').write('foo')
    else:
      script = f"/Applications/PixInsight/src/scripts/WeightedBatchPreprocessing/WeightedBatchPreprocessing.js,automationMode=true,dir={tmp_dir},outputDirectory={tmp_dir},keywords=RIG;INSTRUME;FILTER;FLATVERS"
      print('PixInsight WBPP command line is', script)
      ret = subprocess.run([
          '/Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight',
          '-n', '--automation-mode', f'-r={script}', '--force-exit'
      ]).returncode

    filenames = list(
        str(f)
        for f in Path(src_master_dir).rglob("*mono_drizzle_2x_autocrop.xisf"))
    if not filenames:
      return None

    print('filenames', filenames)
    src_filename = filenames[0]
    dst_filename = os.path.join(
        dst_master_dir,
        f'master_light_{position_string}_{rig}_{filter}_v{version}.xisf')
    shutil.copy(src_filename, dst_filename)
    print('Stacking produced master', src_filename)
    return dst_filename