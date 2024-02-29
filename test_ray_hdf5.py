# yapf: disable
import os
os.environ["ASTROFLOW_MOCK_HDF5"] = "true"

import ray
ray.init(
  address='local',
  namespace='AstroFlow',
  num_cpus=24,
  resources={"HDF5": 1, "PixInsight": 1},
  ignore_reinit_error=True,
  runtime_env={"env_vars": {
    "ASTROFLOW_MOCK_CACHE": "true"
  }},
)

import ray_hdf5
global_var_actor = ray_hdf5.GlobalVarActor.options(name='ray_hdf5',
                                                   namespace='AstroFlow',
                                                   get_if_exists=True).remote()
# yapf: enable

import dataclasses
import unittest


@dataclasses.dataclass
class Foo:
  bar: int = None
  baz: str = None

  def from_hdf5(path):
    return Foo(**ray_hdf5.get_attrs(path))


@ray.remote
def my_remote_fn():
  ray_hdf5.set('hello', 'aloha')


def my_local_fn():
  ray_hdf5.set('ham', 'eggs')


class TestRayHdf5(unittest.TestCase):

  def test_get_set(self):
    # Check behavior when key is missing.
    foo = ray_hdf5.get("foo")
    self.assertEqual(foo, None)

    # Check calling the ray_hdf5 module functions directly.
    ray_hdf5.set('foo', 'bar')
    self.assertEqual(ray_hdf5.get('foo').astype(str), 'bar')

    # Check calling them in a remote task.
    ray.get(my_remote_fn.remote())
    self.assertEqual(ray_hdf5.get('hello').astype(str), 'aloha')

  def test_get_set_dataclass(self):
    foo = Foo(bar=10, baz='banana')
    ray_hdf5.set_attrs('path/to/foo', foo)

    roundtrip_foo = Foo.from_hdf5('path/to/foo')
    self.assertEqual(foo, roundtrip_foo)


if __name__ == "__main__":
  unittest.main()
