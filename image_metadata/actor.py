import ray
import os
from typing import Any
import itertools
import operator
from functools import singledispatch

from cache import cache

if 'ASTROFLOW_MOCK_NAMESPACE' in os.environ:
  namespace = os.environ["ASTROFLOW_MOCK_NAMESPACE"]
else:
  namespace = 'AstroFlow'


@ray.remote(resources={'diskcache': 0.001})
class MetadataActor:

  def __init__(self):
    # TODO: Maybe cache queries?
    # This dict should ONLY store ObjectRefs.
    self.ref_dict = {}

  def get_metadata(
      self,
      input_images,
      predicate_fn,
      transform_fn,
      post_fn,
  ):
    result = []
    for key in cache.iterkeys():
      if type(key) is tuple and key[0] == 'image_metadata':
        if input_images and key[1] not in input_images:
          continue

        image = cache[key]
        if predicate_fn(image):
          result.append(transform_fn(image))

    return post_fn(result)


def get_metadata(input_images=None,
                 predicate_fn=lambda x: x,
                 transform_fn=lambda x: x,
                 post_fn=lambda x: x):
  actor = ray.get_actor('metadata', namespace=namespace)
  return ray.get(
      actor.get_metadata.remote(input_images, predicate_fn, transform_fn,
                                post_fn))


def _group_by_header(input_images, key):
  print(f'### _group_by_header input_images {input_images} key {key}')

  def group_by(images):
    key_fn = operator.itemgetter(1)
    images = sorted(images, key=key_fn)
    return {
        k: set(x[0] for x in g) for k, g in itertools.groupby(images, key=key_fn)
    }

  return get_metadata(input_images=input_images,
                      predicate_fn=lambda i: i.header and key in i.header,
                      transform_fn=lambda i: (i.content_hash, i.header[key]),
                      post_fn=group_by)


@singledispatch
def group_by_header(arg, *args):
  pass


@group_by_header.register
def _(arg: list, *args):
  return _group_by_header(arg, args[0])


@group_by_header.register
def _(arg: set, *args):
  return _group_by_header(arg, args[0])


@group_by_header.register
def _(arg: str):
  return _group_by_header(None, arg)


def _sort_by_header(input_images, key, **kwargs):
  kwargs['key'] = operator.itemgetter(1)
  return get_metadata(input_images=input_images,
                      predicate_fn=lambda i: i.header and key in i.header,
                      transform_fn=lambda i: (i.content_hash, i.header[key]),
                      post_fn=lambda images: sorted(images, **kwargs))


@singledispatch
def sort_by_header(arg, *args, **kwargs):
  pass


@sort_by_header.register
def _(arg: list, *args, **kwargs):
  return _sort_by_header(arg, args[0], **kwargs)


@sort_by_header.register
def _(arg: set, *args, **kwargs):
  return _sort_by_header(arg, args[0], **kwargs)


@sort_by_header.register
def _(arg: str, *args, **kwargs):
  return _sort_by_header(None, arg, **kwargs)


def _cluster_by_header(input_images, key, post_fn):
  return get_metadata(input_images=input_images,
                      predicate_fn=lambda i: i.header and key in i.header,
                      transform_fn=lambda i: (i.content_hash, i.header[key]),
                      post_fn=post_fn)


@singledispatch
def cluster_by_header(arg, *args):
  pass


@cluster_by_header.register
def _(arg: list, *args):
  return _cluster_by_header(arg, args[0], args[1])


@cluster_by_header.register
def _(arg: set, *args):
  return _cluster_by_header(arg, args[0], args[1])


@cluster_by_header.register
def _(arg: str, *args):
  return _cluster_by_header(None, arg, args[0])
