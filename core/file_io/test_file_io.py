from ray_infra import ray_local

import datetime
import numpy as np
import os
import tempfile
import unittest

import file_io.fits_file_io as fits_file_io
import file_io.xisf_file_io as xisf_file_io


class TestFileIO(unittest.TestCase):

  def test_read_fits(self):
    filename = 'testdata/input/Light_Orion_Pane 1_300.0s_Bin1_6200MM_H_20231222-210506_0001.fit'
    header = fits_file_io.read_fits_header(filename, debug=True)

    self.assertEqual(type(header['IMAGETYP']), str)
    self.assertEqual(header['NAXIS'], 2)
    self.assertEqual(header['NAXIS1'], 9576)
    self.assertEqual(header['NAXIS2'], 6388)
    self.assertEqual(type(header['EXPTIME']), float)
    self.assertEqual(type(header['DATE-OBS']), datetime.datetime)

    image = fits_file_io.read_fits_image(filename)
    self.assertEqual(image.shape, (6388, 9576))
    self.assertEqual(image.dtype, np.uint16)

  def test_read_xisf(self):
    filename = 'testdata/output/Stacks/Great Nebula in Orion/masters/master_light_Great Nebula in Orion_RA_82.70_DEC_-5.60_ZWO ASI6200MM Pro_2.21aspp_1.57deg_H_77ade5e8fa458f7f.xisf'
    header = xisf_file_io.read_xisf_header(filename, debug=True)

    self.assertEqual(type(header['IMAGETYP']), str)
    self.assertEqual(header['NAXIS'], 2)
    self.assertEqual(header['NAXIS1'], 19148)
    self.assertEqual(header['NAXIS2'], 12772)
    self.assertEqual(type(header['EXPTIME']), float)
    self.assertEqual(type(header['DATE-OBS']), datetime.datetime)

    image = xisf_file_io.read_xisf_image(filename)
    print(image.shape, image.dtype)
    self.assertEqual(image.shape, (12772, 19148))
    self.assertEqual(image.dtype, np.float32)

  def test_add_xisf_metadata(self):
    input_filename = 'testdata/output/Stacks/Great Nebula in Orion/masters/master_light_Great Nebula in Orion_RA_82.70_DEC_-5.60_ZWO ASI6200MM Pro_2.21aspp_1.57deg_H_77ade5e8fa458f7f.xisf'
    with tempfile.TemporaryDirectory() as temp_dir:
      output_filename = os.path.join(temp_dir, 'foo.xisf')
      xisf_file_io.add_xisf_metadata(input_filename, output_filename, {
          'CONTHASH': 'foo',
          'FLOATVAL': 1.234
      })
      header = xisf_file_io.read_xisf_header(output_filename, debug=True)
      self.assertEqual(header['CONTHASH'], 'foo')
      self.assertEqual(header['FLOATVAL'], 1.234)


if __name__ == '__main__':
  unittest.main()
