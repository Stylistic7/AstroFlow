import tempfile
import os
import shutil

import flat_frame_association
import dark_and_bias_association

TEST_MODE = True


def stack(light_frames, flat_frames):
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
  for config in camera_configs:
    camera, temperature, exposure = config
    bias_frames += dark_and_bias_association.find_bias_frames(
        camera, temperature)
    dark_frames += dark_and_bias_association.find_dark_frames(
        camera, temperature, exposure)

  with tempfile.TemporaryDirectory() as tmp_dir:
    symlink_frames(light_frames, tmp_dir)
    symlink_frames(flat_frames, tmp_dir)
    symlink_frames(bias_frames, tmp_dir)
    symlink_frames(dark_frames, tmp_dir)

    if TEST_MODE:
      master_dir = os.path.join(tmp_dir, 'master')
      os.makedirs(master_dir, exist_ok=True)
      open(os.path.join(master_dir, 'foo'), 'w').write('foo')
    else:
      script = f"/Applications/PixInsight/src/scripts/WeightedBatchPreprocessing/WeightedBatchPreprocessing.js,automationMode=true,dir={tmp_dir},outputDirectory={tmp_dir},keywords=RIG;INSTRUME;FILTER;FLATVERS"
      print('PixInsight WBPP command line is', script)
      ret = subprocess.run([
          '/Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight',
          '-n', '--automation-mode', f'-r={script}', '--force-exit'
      ]).returncode

    output_dir = os.path.join(os.path.commonpath(light_frames), 'master')
    os.makedirs(output_dir, exist_ok=True)
    shutil.copytree(os.path.join(tmp_dir, 'master'),
                    output_dir,
                    dirs_exist_ok=True)