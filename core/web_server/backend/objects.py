import os
import io
import flask
from flask import Blueprint, abort, jsonify

from context import context
from stacks.composite_stack import CompositeStack
from targets.types import Target
from mosaics.types import Mosaic
from products.registry import get_producer, MediaType
from previews.types import QualityLevel

objects_bp = Blueprint("objects", __name__)


@objects_bp.route('/metadata/<object_type>/<object_id>/<filter>')
def metadata(object_type: str, object_id: str, filter: str):
  class_for_type = {'stack': CompositeStack, 'target': Target, 'mosaic': Mosaic}

  # TODO: Unify this with the code in `get_best_image()`.
  if object_type == 'stack':
    object = context.stacks[object_id]
    args = [object]
  elif object_type == 'target':
    object = context.targets[object_id]
    args = [object]
  elif object_type == 'mosaic':
    object = context.mosaics[object_id]
    args = [object, filter]

  levels = {}
  for level in [
      QualityLevel.POST_PROCESSED, QualityLevel.MASTER, QualityLevel.SUBFRAME
  ]:
    if fn := get_producer(MediaType.IMAGE, class_for_type[object_type], level):
      filename = context.output_dir / fn.filename_fn(*args)
      if os.path.exists(filename):
        print(f'### quality_level {level} filename {filename}')
        levels[level.name.lower()] = {'image_path': os.fspath(filename)}

  return jsonify({'levels': levels})


@objects_bp.route('/thumbnail/<object_type>/<object_id>/<filter>')
def thumbnail(object_type: str, object_id: str, filter: str):
  class_for_type = {'stack': CompositeStack, 'target': Target, 'mosaic': Mosaic}

  # TODO: Unify this with the code in `get_best_image()`.
  if object_type == 'stack':
    object = context.stacks[object_id]
    args = [object]
  elif object_type == 'target':
    object = context.targets[object_id]
    args = [object]
  elif object_type == 'mosaic':
    object = context.mosaics[object_id]
    args = [object, filter]

  for level in [
      QualityLevel.POST_PROCESSED, QualityLevel.MASTER, QualityLevel.SUBFRAME
  ]:
    if fn := get_producer(MediaType.IMAGE, class_for_type[object_type], level):
      if hasattr(fn, 'thumbnail_fn'):
        if thumbnail := fn.thumbnail_fn(*args):
          buffer = io.BytesIO()
          buffer.write(thumbnail)
          buffer.seek(0)
          return flask.send_file(buffer, mimetype='image/png')

  return abort(404)


@objects_bp.route('/video/<object_type>/<object_id>/<filter>')
def video(object_type: str, object_id: str, filter: str):
  class_for_type = {'stack': CompositeStack, 'target': Target, 'mosaic': Mosaic}

  # TODO: Unify this with the code in `get_best_image()`.
  if object_type == 'stack':
    object = context.stacks[object_id]
    args = [object]
  elif object_type == 'target':
    object = context.targets[object_id]
    args = [object]
  elif object_type == 'mosaic':
    object = context.mosaics[object_id]
    args = [object, filter]

  for level in [
      QualityLevel.POST_PROCESSED, QualityLevel.MASTER, QualityLevel.SUBFRAME
  ]:
    if fn := get_producer(MediaType.VIDEO, class_for_type[object_type], level):
      if hasattr(fn, 'video_fn'):
        if video := fn.video_fn(*args):
          buffer = io.BytesIO()
          buffer.write(video)
          buffer.seek(0)
          # TODO: Support other codecs.
          return flask.send_file(buffer, mimetype='video/webm')

  return abort(404)


@objects_bp.route("/image/<object_type>/<object_id>")
def image(object_type, object_id, z, x, y):
  return abort(404)
