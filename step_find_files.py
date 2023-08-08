from pathlib import Path
import file_io
import sys
import os
import itertools


def run(input_dir, max_frames=None):
  filenames = list(str(f) for f in Path(input_dir).rglob("*.fit"))
  print(f'Found {len(filenames)} FITS files')
  filenames = [
      f for f in filenames
      if not os.path.basename(f).lower().startswith('master')
  ]
  print(f'Found {len(filenames)} FITS files that are not masters')
  filenames = [
      f for f in filenames
      if os.path.split(os.path.dirname(f))[1].lower() != 'bad'
  ]
  print(f'Found {len(filenames)} FITS files that are not marked bad')

  def image_type(f):
    return file_io.read_fits(f)['IMAGETYP']

  filenames = sorted(filenames, key=image_type)
  files = {k: list(g) for k, g in itertools.groupby(filenames, key=image_type)}

  light_frames = files['Light'][0:max_frames] if 'Light' in files else []
  dark_frames = files['Dark'][0:max_frames] if 'Dark' in files else []
  bias_frames = files['Bias'][0:max_frames] if 'Bias' in files else []
  flat_frames = files['Flat'][0:max_frames] if 'Flat' in files else []

  return light_frames, dark_frames, bias_frames, flat_frames
