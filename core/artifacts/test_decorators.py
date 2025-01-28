from ray_infra import ray_local

import numpy as np
import os
import unittest
import ray
from typing import Any
import astropy.io.fits as pyfits  # type: ignore

from artifacts import Artifact, FileArtifact, CacheArtifact
from testing.test_context import make_test_context
from context import context
from artifacts.decorators import saves_artifacts, should_run
from hashing import hash_list


# This function should not care about where the final output directory is;
# it is only concerned with the relative path inside that dir.
def foo_artifacts(mosaic_id: str, filter: str) -> list[Artifact]:
  filename = os.path.join(context.output_dir,
                          f'master_light_{mosaic_id}_{filter}.fits')
  content_hash = hash_list([mosaic_id, filter])
  return [FileArtifact(filename, content_hash)]


@ray.remote
@saves_artifacts(foo_artifacts)
def foo(mosaic_id: str, filter: str) -> list[Artifact]:
  print('### Running foo')
  context.num_foo_runs += 1
  output_artifacts = foo_artifacts(mosaic_id, filter)
  output_artifact = next(iter(output_artifacts))
  assert isinstance(output_artifact, FileArtifact)
  image = np.random.random((10, 10))
  hdu = pyfits.PrimaryHDU(image)
  hdu.header['CONTHASH'] = output_artifact.content_hash
  hdu.writeto(output_artifact.filename)

  return output_artifacts


def bar_artifacts(mosaic_id: str, filter: str) -> list[Artifact]:
  return [CacheArtifact(key=('bar', mosaic_id, filter), content_hash='bar')]


@ray.remote
@saves_artifacts(bar_artifacts)
def bar(mosaic_id: str, filter: str) -> list[Artifact]:
  print('### Running bar')
  context.num_bar_runs += 1
  output_artifacts = bar_artifacts(mosaic_id, filter)
  output_artifact = next(iter(output_artifacts))
  assert isinstance(output_artifact, CacheArtifact)
  context[output_artifact.key] = 'bar'
  return output_artifacts


def local_fn_artifacts(mosaic_id: str, filter: str) -> list[Artifact]:
  return [
      CacheArtifact(key=('local_fn', mosaic_id, filter), content_hash='foo')
  ]


@saves_artifacts(local_fn_artifacts)
def local_fn(mosaic_id: str, filter: str) -> list[Artifact]:
  print('### Running local_fn')
  context.num_local_fn_runs += 1
  output_artifacts = local_fn_artifacts(mosaic_id, filter)
  output_artifact = next(iter(output_artifacts))
  assert isinstance(output_artifact, CacheArtifact)
  context[output_artifact.key] = 'local_fn'
  return output_artifacts


class TestDecorators(unittest.TestCase):

  def setUp(self) -> None:
    self.input_dir_obj, self.output_dir_obj = make_test_context()
    context.num_foo_runs = 0
    context.num_bar_runs = 0
    context.num_local_fn_runs = 0

  def tearDown(self) -> None:
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_saves_files(self) -> None:
    self.assertTrue(should_run(foo_artifacts, 'abc123', 'H'))

    ray.get(foo.remote('abc123', 'H'))
    self.assertFalse(should_run(foo_artifacts, 'abc123', 'H'))

  def test_saves_cache_keys(self) -> None:
    self.assertTrue(should_run(bar_artifacts, 'abc123', 'H'))

    ray.get(bar.remote('abc123', 'H'))
    self.assertFalse(should_run(bar_artifacts, 'abc123', 'H'))

  def test_local_fn(self) -> None:
    local_fn('abc123', 'H')
    local_fn('abc123', 'H')
    local_fn('abc123', 'H')
    local_fn('abc123', 'H')

    self.assertEqual(context.num_local_fn_runs, 1)

  def test_foo_runs_only_once(self) -> None:
    ray.get(foo.remote('abc123', 'H'))
    ray.get(foo.remote('abc123', 'H'))
    ray.get(foo.remote('abc123', 'H'))
    ray.get(foo.remote('abc123', 'H'))

    self.assertEqual(context.num_foo_runs, 1)


if __name__ == '__main__':
  unittest.main()
