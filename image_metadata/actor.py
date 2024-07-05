import ray
import os
from typing import Any
import itertools
import operator
from functools import singledispatch
from collections.abc import Iterable

from cache import cache
from context import context

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
        if input_images is not None and key[1] not in input_images:
          continue

        image = cache[key]
        if predicate_fn(image):
          result.append(transform_fn(image))

    return post_fn(result)


# TODO: Maybe implement a version of this that accepts a single str.
def get_metadata(input_images=None,
                 predicate_fn=lambda x: True,
                 transform_fn=lambda x: x,
                 post_fn=lambda x: x):
  if input_images is not None and len(input_images) == 0:
    return []

  actor = ray.get_actor('metadata', namespace=namespace)
  return ray.get(
      actor.get_metadata.remote(input_images, predicate_fn, transform_fn,
                                post_fn))


# TODO: Maybe unify API with @singledispatch.
def get_metadata_for_stack(stack_id):
  stacks = context.stacks
  stack = stacks[stack_id]

  # Start with all of the stacks included in this stack.
  hashes = stack.all_frames()

  settings_key = ('stack_settings', stack_id)
  settings = context[settings_key] or {}
  dependencies = stack.dependencies | settings

  # Include all the frames from all of the stacks' dependencies (which are
  # typically calibration stacks).)
  if 'bias' in dependencies:
    hashes.extend(stacks[dependencies['bias']].frames)
  if 'dark' in dependencies:
    hashes.extend(stacks[dependencies['dark']].frames)

  # Include every known flat calibration stack, just for completeness.
  for s in stacks.values():
    if (not s.stable_hash and s.image_type == 'flat' and
        s.camera == stack.camera and s.filter == stack.filter):
      hashes.extend(s.frames)

  return get_metadata(
      hashes, post_fn=lambda images: {i.content_hash: i for i in images})


def _group_by_header(input_images, key):

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


@group_by_header.register(Iterable)
def _(frames, *args):
  return _group_by_header(frames, args[0])


@group_by_header.register(str)
def _(key):
  return _group_by_header(None, key)


def _sort_by_header(input_images, key, **kwargs):
  kwargs['key'] = operator.itemgetter(1)
  return get_metadata(
      input_images=input_images,
      predicate_fn=lambda i: i.header and key in i.header,
      transform_fn=lambda i: (i.content_hash, i.header[key]),
      post_fn=lambda images: [i[0] for i in sorted(images, **kwargs)])


@singledispatch
def sort_by_header(arg, *args, **kwargs):
  pass


@sort_by_header.register(Iterable)
def _(frames, *args, **kwargs):
  return _sort_by_header(frames, args[0], **kwargs)


@sort_by_header.register(str)
def _(key, *args, **kwargs):
  return _sort_by_header(None, key, **kwargs)


def _cluster_by_header(input_images, key, post_fn):
  return get_metadata(input_images=input_images,
                      predicate_fn=lambda i: i.header and key in i.header,
                      transform_fn=lambda i: (i.content_hash, i.header[key]),
                      post_fn=post_fn)


@singledispatch
def cluster_by_header(arg, *args):
  pass


@cluster_by_header.register(Iterable)
def _(frames, *args):
  return _cluster_by_header(frames, args[0], args[1])


@cluster_by_header.register(str)
def _(key, *args):
  return _cluster_by_header(None, key, args[0])
