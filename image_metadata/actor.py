import ray
import os
from typing import Any
import itertools
import operator

from cache import cache

if 'ASTROFLOW_MOCK_NAMESPACE' in os.environ:
  namespace = os.environ["ASTROFLOW_MOCK_NAMESPACE"]
else:
  namespace = 'AstroFlow'

print(f'### MetadataActor using namespace {namespace}')


@ray.remote(resources={'diskcache': 0.001})
class MetadataActor:

  def __init__(self):
    # TODO: Maybe cache queries?
    # This dict should ONLY store ObjectRefs.
    self.ref_dict = {}

  def get_metadata(self, predicate_fn, transform_fn, post_fn):
    result = []
    for key in cache.iterkeys():
      if type(key) is tuple and key[0] == 'image_metadata':
        image = cache[key]
        if predicate_fn(image):
          result.append(transform_fn(image))

    return post_fn(result)


class Metadata:

  def get_metadata(self,
                   predicate_fn=lambda x: x,
                   transform_fn=lambda x: x,
                   post_fn=lambda x: x):
    actor = ray.get_actor('metadata', namespace=namespace)
    return ray.get(
        actor.get_metadata.remote(predicate_fn, transform_fn, post_fn))

  def group_by_header(self, key):

    def group_by(images):
      key_fn = operator.itemgetter(1)
      images = sorted(images, key=key_fn)
      return {
          k: set(x[0] for x in g)
          for k, g in itertools.groupby(images, key=key_fn)
      }

    return self.get_metadata(
        predicate_fn=lambda i: i.header and key in i.header,
        transform_fn=lambda i: (i.content_hash, i.header[key]),
        post_fn=group_by)


metadata = Metadata()
