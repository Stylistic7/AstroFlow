import functools
from pathlib import Path
from PIL import Image
from typing import Callable, Optional, Union

from context import context
from previews.operations import resize_array, array_to_png, autostretch
from file_io.generic import read_image
from stacks.simple_stack import SimpleStack
from stacks.composite_stack import CompositeStack
from attributes import ChronologyAttributes
from products.registry import register


def _get_type_name(cls: type) -> str:
  type_to_name = {
      SimpleStack: "stack",
      CompositeStack: "stack",
  }

  if cls in type_to_name:
    return type_to_name[cls]
  else:
    return cls.__name__.lower()


# I need to take a step back and figure out what I'm actually trying to
# accomplish.
#
# I want the UI to be able to show users the highest-quality version of their
# images. Stacks, targets, and mosaics can all be shown at subframe, master,
# or post-processed quality, and there may be more quality levels added later.
#
# The web server just knows assets by their type and id, like
# ('mosaic', 'abc123').
#
# For any asset, multiple levels may exist. Each is computed by some kind of
# image-processing function. Some of the images are huge, hundreds of megabytes
# or even tens or hundreds of gigabytes. They all need thumbnails for cards,
# and then tiled TIFFs for viewing at full resolution.
#
# The web server shouldn't care about the specific functions that generate the
# images; that may change over time. Instead, for a given object, it should
# look up all of the versions (quality levels and their thumbnails) and then
# automatically display the best one.
#
# In addition, when images are unavailable (e.g. they're on an external drive
# that is not connected), the UI should automatically fall back on using
# lower-quality versions.
#
# It's clear that thumbnails should always be stored in the cache, so they're
# available even when disks are disconnected.


def path_template(template: str) -> Callable:
  return lambda obj: Path(template.format(**vars(obj)))


def register_image_producer(
    object_type: type,
    level,
    filename_fn: Optional[Union[str, Callable[..., Path]]] = None):

  if isinstance(filename_fn, str):
    filename_fn = path_template(filename_fn)

  def decorator(func: Callable[..., Path]):
    if filename_fn is not None:
      func.filename_fn = filename_fn

    register(object_type, level, func)

    return func

  return decorator
