import unittest
import numpy as np
import tempfile

from cache import cache
from stack import Stack
from target import Target
from mosaic import Mosaic
from make_test_frames import make_test_frames
import step_mosaic
from mosaics import identification
from stack_identification import identify_stacks
from astro_flow_direct import group_stacks_into_targets
from spatial_association import find_components_by_cosine_similarity


class TestMosaics(unittest.TestCase):
  def test_four_panels_on_equator(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      print('### 1')
      image_hashes, fits_headers, plate_solutions = make_test_frames(
          number=10,
          prefix='foo',
          project_dir=tmp_dir,
          ra=179,
          dec=0,
          rotation_angle_deg=0,
          dither=0,
          fill_value=50)
      print('### 2')
      image_hashes, fits_headers, plate_solutions = make_test_frames(
          number=10,
          prefix='bar',
          project_dir=tmp_dir,
          ra=181,
          dec=0,
          rotation_angle_deg=0,
          dither=0,
          fill_value=100,
          image_hashes=image_hashes,
          fits_headers=fits_headers,
          plate_solutions=plate_solutions)
      print('### 3')
      image_hashes, fits_headers, plate_solutions = make_test_frames(
          number=10,
          prefix='baz',
          project_dir=tmp_dir,
          ra=179,
          dec=1,
          rotation_angle_deg=0,
          dither=0,
          fill_value=150,
          image_hashes=image_hashes,
          fits_headers=fits_headers,
          plate_solutions=plate_solutions)
      print('### 4')
      image_hashes, fits_headers, plate_solutions = make_test_frames(
          number=10,
          prefix='qux',
          project_dir=tmp_dir,
          ra=181,
          dec=1,
          rotation_angle_deg=0,
          dither=0,
          fill_value=200,
          image_hashes=image_hashes,
          fits_headers=fits_headers,
          plate_solutions=plate_solutions)

      spatial_components = find_components_by_cosine_similarity(
          image_hashes.keys(), plate_solutions, fits_headers)
      stacks = identify_stacks(spatial_components, plate_solutions,
                               fits_headers)
      stacks = {s.id: s for s in stacks}

      targets = group_stacks_into_targets(stacks, plate_solutions,
                                          fits_headers)

      mosaics = identification.identify_mosiacs(targets)
      self.assertEqual(len(mosaics), 1)
      mosaic = mosaics[0]

      step_mosaic.process_mosaic(image_hashes,
                                 plate_solutions,
                                 mosaic,
                                 targets,
                                 stacks,
                                 preview=True,
                                 output_dir=tmp_dir)

      self.assertTrue(('mosaic_preview_image', mosaic.id) in cache)


if __name__ == "__main__":
  unittest.main()