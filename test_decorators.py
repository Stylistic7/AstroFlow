import ray_local

import numpy as np
import os
import unittest
import ray
import astropy.io.fits as pyfits  # type: ignore

from artifact import Artifact
from make_test_context import make_test_context
from context import context
from decorators import saves_artifacts
from hashing import hash_list


# This function should not care about where the final output directory is;
# it is only concerned with the relative path inside that dir.
def foo_artifacts(mosaic_id, filter):
  content_hash = hash_list([mosaic_id, filter])
  return [
      Artifact(content_hash=content_hash,
               filename=f'master_light_{mosaic_id}_{filter}.fit')
  ]


@ray.remote
@saves_artifacts(foo_artifacts, in_test=True, debug=True)
def foo(mosaic_id, filter):
  print('### Running foo')
  artifact = next(iter(foo_artifacts(mosaic_id, filter)))
  filename = os.path.join(context.output_dir,
                          f'master_light_{mosaic_id}_{filter}.fit')
  image = np.random.random((10, 10))
  hdu = pyfits.PrimaryHDU(image)
  hdu.header['CONTHASH'] = artifact.content_hash
  hdu.writeto(filename)

  return filename


def bar_artifacts(mosaic_id, filter):
  return [Artifact(key=('bar', mosaic_id, filter))]


@ray.remote
@saves_artifacts(bar_artifacts, in_test=True, debug=True)
def bar(mosaic_id, filter):
  print('### Running bar')
  key = next(iter(bar_artifacts(mosaic_id, filter))).key
  context[key] = 'bar'


class TestDecorators(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()

  def tearDown(self):
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_saves_files(self):
    self.assertTrue(foo._function.should_run('abc123', 'H'))

    ray.get(foo.remote('abc123', 'H'))
    self.assertFalse(foo._function.should_run('abc123', 'H'))

  def test_saves_cache_keys(self):
    self.assertTrue(bar._function.should_run('abc123', 'H'))

    ray.get(bar.remote('abc123', 'H'))
    self.assertFalse(bar._function.should_run('abc123', 'H'))


if __name__ == '__main__':
  unittest.main()
