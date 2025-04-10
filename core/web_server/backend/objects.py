import os
from flask import Blueprint, abort, jsonify

from context import context
from stacks.composite_stack import CompositeStack
from targets.types import Target
from mosaics.types import Mosaic
from products.registry import get_image_producer
from previews import QualityLevel

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

  result = {'levels': {}}
  for level in [
      QualityLevel.SUBFRAME, QualityLevel.MASTER, QualityLevel.POST_PROCESSED
  ]:
    fn = get_image_producer(class_for_type[object_type], level)
    if fn is None:
      continue

    filename = context.output_dir / fn.filename_fn(*args)
    print(f'### quality_level {level} filename {filename}')

    result['levels'][level.name.lower()] = {'image_path': os.fspath(filename)}

  return jsonify(result)


@objects_bp.route("/image/<object_type>/<object_id>")
def image(object_type, object_id, z, x, y):
  return abort(404)
