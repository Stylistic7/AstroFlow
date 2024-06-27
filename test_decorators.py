import ray_local

import numpy as np
import os
import unittest
import ray
import astropy.io.fits as pyfits  # type: ignore

from make_test_context import make_test_context
from context import context
from decorators import saves_artifacts
from hashing import hash_list


# This function should not care about where the final output directory is;
# it is only concerned with the relative path inside that dir.
def foo_artifacts(mosaic_id, filter):
  content_hash = hash_list([mosaic_id, filter])
  return {content_hash: f'master_light_{mosaic_id}_{filter}.fit'}


@saves_artifacts(foo_artifacts)
@ray.remote
def foo(mosaic_id, filter):
  content_hash, filename = next(iter(foo_artifacts(mosaic_id, filter).items()))
  filename = os.path.join(context.output_dir,
                          f'master_light_{mosaic_id}_{filter}.fit')
  image = np.random.random((10, 10))
  hdu = pyfits.PrimaryHDU(image)
  hdu.header['CONTHASH'] = content_hash
  hdu.writeto(filename)

  return filename


def bar_artifacts(mosaic_id, filter):
  return [('bar', mosaic_id, filter)]


@saves_artifacts(bar_artifacts)
@ray.remote
def bar(mosaic_id, filter):
  key = next(iter(bar_artifacts(mosaic_id, filter)))
  context[key] = 'bar'


def baz_artifacts(mosaic_id, filter):
  return [('baz', mosaic_id, filter)]


@saves_artifacts(baz_artifacts)
def baz(mosaic_id, filter):
  key = next(iter(baz_artifacts(mosaic_id, filter)))
  context[key] = 'baz'


class TestDecorators(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()

  def test_saves_files(self):
    self.assertTrue(foo.should_run('abc123', 'H'))

    ray.get(foo.remote('abc123', 'H'))
    self.assertFalse(foo.should_run('abc123', 'H'))

  def test_saves_cache_keys(self):
    self.assertTrue(bar.should_run('abc123', 'H'))

    ray.get(bar.remote('abc123', 'H'))
    self.assertFalse(bar.should_run('abc123', 'H'))

  def test_local_function(self):
    self.assertTrue(baz.should_run('abc123', 'H'))

    baz('abc123', 'H')
    self.assertFalse(baz.should_run('abc123', 'H'))


if __name__ == '__main__':
  unittest.main()
