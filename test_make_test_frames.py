# yapf: disable
import os
os.environ["ASTROFLOW_MOCK_CACHE"] = "true"

from cache import cache

import ray
ray.init(
  address="local",
  resources={"diskcache": 1},
  ignore_reinit_error=True,
  runtime_env={"env_vars": {
    "ASTROFLOW_MOCK_CACHE": "true"
  }},
)
# yapf: enable

import numpy as np
import unittest
import tempfile
import datetime

from spherical_geometry.polygon import SphericalPolygon
import astropy

from plate_solving import PlateSolution
import make_test_frames


class TestMakeTestFrames(unittest.TestCase):

  def test_one_perfectly_centered_image(self):
    image_hashes, fits_headers, plate_solutions = make_test_frames.make_test_frames(
        number=1, prefix='foo', dither=0)

    expected_fits_headers = {
        'foo_00000.fit': {
            'DATE-OBS': datetime.datetime(2023, 12, 6, 16, 29, 43),
            'IMAGETYP': 'Light',
            'EXPOSURE': 300,
            'CRVAL1': 100.0,
            'CRVAL2': 45.0
        }
    }

    expected_plate_solutions = {
        'foo_00000.fit':
            PlateSolution(image_filename='foo_00000.fit',
                          header=None,
                          num_stars=0,
                          center=(100.0, 45.0),
                          image_scale=None,
                          rotation_angle=None,
                          footprint=[(98.0, 46.0), (98.0, 44.0), (102.0, 44.0),
                                     (102.0, 46.0)])
    }

    # TODO: Actually compare these...

  def test_testdata_images(self):
    image_hashes, fits_headers, plate_solutions = make_test_frames.make_test_frames(
        number=7,
        prefix='bubble_nebula',
        dither=0,
        testdata_image='bubble_nebula.npy')

    image_hashes, fits_headers, plate_solutions = make_test_frames.make_test_frames(
        number=11,
        prefix='sadr_region',
        dither=0,
        testdata_image='sadr_region.npy',
        image_hashes=image_hashes,
        fits_headers=fits_headers,
        plate_solutions=plate_solutions)

    self.assertEqual(len(image_hashes), 18)


if __name__ == '__main__':
  unittest.main()
