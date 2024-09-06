import ray
import os
from typing import Any
import itertools
import operator
from functools import singledispatch
from collections.abc import Iterable

from context import get_cache
from context import context
from stacks import hashes_for_stack

# TODO: Rename this file.


@ray.remote(resources={'diskcache': 0.001})
def get_metadata_from_diskcache(
    input_images,
    predicate_fn,
    transform_fn,
    post_fn,
):
  cache = get_cache()

  result = []

  # TODO: Rename input_image to hash.
  def iterate_over_input_images():
    for input_image in input_images:
      key = ('image_metadata', input_image)

      image = cache.get(key, None)
      assert image is not None, f'Could not get image metadata for hash {input_image}'

      if predicate_fn(image):
        result.append(transform_fn(image))

    print(f'### get_metadata {len(input_images)} requested, '
          f'{len(result)} returned')

    return post_fn(result)

  def iterate_over_all_keys():
    num_keys = 0
    for key in cache.iterkeys():
      if type(key) is tuple and key[0] == 'image_metadata':
        num_keys += 1
        image = cache[key]
        if predicate_fn(image):
          result.append(transform_fn(image))

    print(f'### get_metadata {num_keys} keys, all requested, '
          f'{len(result)} returned')

    return post_fn(result)

  if input_images is not None:
    return iterate_over_input_images()
  else:
    return iterate_over_all_keys()


def get_metadata(input_images=None,
                 predicate_fn=lambda x: True,
                 transform_fn=lambda x: x,
                 post_fn=lambda x: x):
  if input_images is not None:
    input_images = tuple(input_images)
    assert len(input_images) > 0

  return _get_metadata(input_images, predicate_fn, transform_fn, post_fn)


# TODO: Maybe implement a version of this that accepts a single str.
def _get_metadata(input_images=None,
                  predicate_fn=lambda x: True,
                  transform_fn=lambda x: x,
                  post_fn=lambda x: x):
  if input_images is not None and len(input_images) == 0:
    return []

  return ray.get(
      get_metadata_from_diskcache.remote(input_images, predicate_fn,
                                         transform_fn, post_fn))


# TODO: Maybe unify API with @singledispatch.
def get_metadata_for_mosaic(mosaic_id):
  targets = context.targets
  mosaics = context.mosaics

  mosaic = mosaics[mosaic_id]

  hashes = set()
  for target_id in mosaic.target_ids:
    for stack_id in targets[target_id].stack_ids:
      hashes.update(hashes_for_stack(stack_id))

  return get_metadata(
      hashes, post_fn=lambda images: {i.content_hash: i for i in images})


# TODO: Maybe unify API with @singledispatch.
def get_metadata_for_target(target_id):
  targets = context.targets

  hashes = set()
  for stack_id in targets[target_id].stack_ids:
    hashes.update(hashes_for_stack(stack_id))

  return get_metadata(
      hashes, post_fn=lambda images: {i.content_hash: i for i in images})


# TODO: Maybe unify API with @singledispatch.
def get_metadata_for_stack(stack_id):
  hashes = hashes_for_stack(stack_id)
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
