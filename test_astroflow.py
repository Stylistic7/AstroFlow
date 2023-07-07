import unittest
import tempfile

import file_io
import make_test_frames
import spatial_association


class TestAstroFlow(unittest.TestCase):
  def test_spatial_association(self):
    with tempfile.TemporaryDirectory() as project_dir:
      # Make one group of light frames at one well-defined position in the sky.
      make_test_frames.make_light_frames(project_dir, 311, 37, 0.05)

      # Make another group of light frames a few degrees away.
      make_test_frames.make_light_frames(project_dir, 311, 40, 0.05)

      fits_filenames = file_io.find_files(project_dir, 'Light*.fit')
      print('fits_filenames', fits_filenames)

      positions = []
      for filename in fits_filenames:
        header = file_io.read_fits(filename)
        positions.append([header['CRVAL1'], header['CRVAL2']])

      print('positions', positions)
      pred = list(spatial_association.find_components(positions))
      print('pred', pred)

      self.assertEqual(pred.count(0), 10)
      self.assertEqual(pred.count(1), 10)


if __name__ == '__main__':
  unittest.main()
