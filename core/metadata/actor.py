import ray
import os
from typing import Any
import itertools
import operator
from functools import singledispatch
from collections.abc import Iterable

from context import get_cache
from context import context

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
