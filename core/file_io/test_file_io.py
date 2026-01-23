import datetime
import numpy as np
import os
import tempfile
import unittest
import shutil
from pathlib import Path

from file_io.types import FitsFilePath, XisfFilePath, WithSidecar
from file_io.generic import read_header, read_image, update_header, get_content_hash, set_content_hash, copy_image
from file_io.sidecar import sidecar_path_for

class TestFileIO(unittest.TestCase):

  def test_read_fits(self):
    filename = FitsFilePath(
        'testdata/input/Light_Orion_Pane 1_300.0s_Bin1_6200MM_H_20231222-210506_0001.fit'
    )
    header = read_header(filename)

    self.assertEqual(type(header['IMAGETYP']), str)
    self.assertEqual(header['NAXIS'], 2)
    self.assertEqual(header['NAXIS1'], 9576)
    self.assertEqual(header['NAXIS2'], 6388)
    self.assertEqual(type(header['EXPTIME']), float)
    self.assertEqual(type(header['DATE-OBS']), datetime.datetime)

    image = read_image(filename)
    self.assertEqual(image.shape, (6388, 9576))
    self.assertEqual(image.dtype, np.uint16)

  def test_read_xisf(self):
    filename = XisfFilePath(
        'testdata/output/Stacks/Great Nebula in Orion/masters/master_light_Great Nebula in Orion_RA_82.70_DEC_-5.60_ZWO ASI6200MM Pro_2.21aspp_1.57deg_H_77ade5e8fa458f7f.xisf'
    )
    header = read_header(filename)

    self.assertEqual(type(header['IMAGETYP']), str)
    self.assertEqual(header['NAXIS'], 2)
    self.assertEqual(header['NAXIS1'], 19148)
    self.assertEqual(header['NAXIS2'], 12772)
    self.assertEqual(type(header['EXPTIME']), float)
    self.assertEqual(type(header['DATE-OBS']), datetime.datetime)

    image = read_image(filename)
    print(image.shape, image.dtype)
    self.assertEqual(image.shape, (12772, 19148))
    self.assertEqual(image.dtype, np.float32)

  def test_add_xisf_metadata(self):
    input_filename = XisfFilePath(
        'testdata/output/Stacks/Great Nebula in Orion/masters/master_light_Great Nebula in Orion_RA_82.70_DEC_-5.60_ZWO ASI6200MM Pro_2.21aspp_1.57deg_H_77ade5e8fa458f7f.xisf'
    )
    with tempfile.TemporaryDirectory() as temp_dir:
      output_filename = XisfFilePath(os.path.join(temp_dir, 'foo.xisf'))
      shutil.copy(input_filename, output_filename)
      update_header(output_filename, {'CONTHASH': 'foo', 'FLOATVAL': 1.234})
      header = read_header(output_filename)
      self.assertEqual(header['CONTHASH'], 'foo')
      self.assertEqual(header['FLOATVAL'], 1.234)

  def test_sidecar(self):
    input_filename = Path(
        'testdata/output/Stacks/Great Nebula in Orion/masters/master_light_Great Nebula in Orion_RA_82.70_DEC_-5.60_ZWO ASI6200MM Pro_2.21aspp_1.57deg_H_77ade5e8fa458f7f.xisf'
    )
    with tempfile.TemporaryDirectory() as temp_dir:
      output_file = WithSidecar(XisfFilePath(os.path.join(temp_dir,
                                                          'foo.xisf')))
      shutil.copy(input_filename, output_file.path)
      set_content_hash(output_file, 'foo')

      sidecar_path = sidecar_path_for(output_file.path)
      self.assertTrue(os.path.exists(sidecar_path))

      update_header(output_file, {'FLOATVAL': 1.234})
      header = read_header(output_file)
      self.assertEqual(header['FLOATVAL'], 1.234)

      self.assertEqual(get_content_hash(output_file), 'foo')

  def test_copy(self):
    input_filename = Path(
        'testdata/output/Stacks/Great Nebula in Orion/masters/master_light_Great Nebula in Orion_RA_82.70_DEC_-5.60_ZWO ASI6200MM Pro_2.21aspp_1.57deg_H_77ade5e8fa458f7f.xisf'
    )
    with tempfile.TemporaryDirectory() as temp_dir:
      first_file = WithSidecar(XisfFilePath(os.path.join(temp_dir,
                                                          'foo.xisf')))
      shutil.copy(input_filename, first_file.path)
      set_content_hash(first_file, 'foo')
      update_header(first_file, {'FLOATVAL': 1.234})

      second_file = WithSidecar(XisfFilePath(os.path.join(temp_dir,
                                                          'bar.xisf')))
      copy_image(first_file, second_file)

      header = read_header(second_file)
      self.assertEqual(header['FLOATVAL'], 1.234)

if __name__ == '__main__':
  unittest.main()
