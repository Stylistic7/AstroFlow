import ray
import h5py
import dataclasses
import os
import tempfile
import numpy as np


@ray.remote(resources={'HDF5': 0.001})
class GlobalVarActor:

  def __init__(self):
    if 'ASTROFLOW_MOCK_HDF5' in os.environ:
      self.temp_file = tempfile.NamedTemporaryFile()
      self.file = h5py.File(self.temp_file.name, 'w')
    else:
      self.file = h5py.File('AstroFlow.h5', 'a')

  def get(self, key, default=None):
    result = np.asarray(self.file.get(key, default))
    return result

  def get_attrs(self, key, default=None):
    if default:
      default = {}

    if key not in self.file:
      return default

    return dict(self.file[key].attrs)

  def set(self, key, value):
    self.file[key] = value

  def set_attrs(self, key, value):
    if key not in self.file:
      self.file.create_group(key)

    if dataclasses.is_dataclass(value):
      self.file[key].attrs.update(dataclasses.asdict(value))
    else:
      self.file[key.attrs.update(value)]

    return value

  def contains(self, key):
    return key in self.file


def get(key, default=None):
  actor = ray.get_actor('ray_hdf5', namespace='AstroFlow')
  return ray.get(actor.get.remote(key, default))


def get_attrs(key, default=None):
  actor = ray.get_actor('ray_hdf5', namespace='AstroFlow')
  return ray.get(actor.get_attrs.remote(key, default))


def set(key, value):
  actor = ray.get_actor('ray_hdf5', namespace='AstroFlow')
  return ray.get(actor.set.remote(key, value))


def set_attrs(key, value):
  actor = ray.get_actor('ray_hdf5', namespace='AstroFlow')
  return ray.get(actor.set_attrs.remote(key, value))


def contains(key):
  actor = ray.get_actor('ray_hdf5', namespace='AstroFlow')
  return ray.get(actor.contains.remote(key))
