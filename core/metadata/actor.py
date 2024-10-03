import ray
from typing import Any, Callable

from context import get_cache
from metadata.classes import FrameHash, ImageMetadata

# TODO: Rename this file.


@ray.remote(resources={'diskcache': 0.001})
def get_metadata_from_diskcache(
    input_images: list[FrameHash] | None,
    predicate_fn: Callable[[ImageMetadata], bool],
    transform_fn: Callable[[ImageMetadata], Any],
    post_fn: Callable[[list[ImageMetadata]], Any],
) -> Any:
  cache = get_cache()

  result = []

  # TODO: Rename input_image to hash.
  def iterate_over_input_images() -> Any:
    assert input_images is not None
    for input_image in input_images:
      key = ('image_metadata', input_image)

      image = cache.get(key, None)
      assert image is not None, f'Could not get image metadata for hash {input_image}'

      if predicate_fn(image):
        result.append(transform_fn(image))

    print(f'### get_metadata {len(input_images)} requested, '
          f'{len(result)} returned')

    return post_fn(result)

  def iterate_over_all_keys() -> Any:
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


def get_metadata(
    input_images: list[FrameHash] | None = None,
    predicate_fn: Callable[[ImageMetadata], bool] = lambda x: True,
    transform_fn: Callable[[ImageMetadata], Any] = lambda x: x,
    post_fn: Callable[[list[ImageMetadata]], Any] = lambda x: x) -> Any:
  return ray.get(
      get_metadata_from_diskcache.remote(input_images, predicate_fn,
                                         transform_fn, post_fn))
