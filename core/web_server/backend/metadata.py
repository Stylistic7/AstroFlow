import os
from flask import Blueprint, jsonify, abort
from typing import Any

from context import context
from stacks.simple_stack import CalibrationStack
from stacks.composite_stack import CompositeStack
from targets.types import Target
from mosaics.types import Mosaic, Viewbox
from mosaics.viewbox import compute_viewbox
from products.registry import get_producer, MediaType
from previews.types import QualityLevel
from mosaics.trimming import project_to_viewbox

metadata_bp = Blueprint("metadata", __name__)


def get_metadata(object_type: type, *args):
  for level in [
      QualityLevel.POST_PROCESSED, QualityLevel.MASTER, QualityLevel.SUBFRAME
  ]:
    image_path = None
    if fn := get_producer(MediaType.IMAGE, object_type, level):
      filename = fn.filename_fn(*args).resolve_at(context.output_dir)
      if os.path.exists(filename):
        image_path = os.fspath(filename)

    if image_path is None:
      continue

    result = {'image_path': image_path}

    if object_type is CalibrationStack:
      return result

    obj = args[0]
    viewbox = compute_viewbox(level, obj)  # FIXME
    objects = []

    # TODO: What do you call an object with astrometric and chronology attributes?
    def project(object: Any, viewbox: Viewbox) -> dict[str, Any]:
      polygon = project_to_viewbox(object.footprint, viewbox)
      return {
          'type': type(object).__name__.lower(),
          'color': '0xff0000',
          'stable_hash': object.stable_hash,
          'points': list(polygon.exterior.coords)
      }

    stacks = context.stacks
    targets = context.targets

    objects.append(project(obj, viewbox))
    if isinstance(obj, Target):
      objects += [project(stacks[s], viewbox) for s in obj.stack_ids]
    if isinstance(obj, Mosaic):
      objects += [project(targets[t], viewbox) for t in obj.target_ids]

    result['objects'] = objects

    return jsonify(result)

  return abort(404)


@metadata_bp.route('/metadata/stack/<stack_id>')
def stack_metadata(stack_id: str):
  return get_metadata(CompositeStack, context.stacks[stack_id])


@metadata_bp.route('/metadata/calibration_stack/<stack_id>')
def calibration_stack_metadata(stack_id: str):
  return get_metadata(CalibrationStack, context.calibration_stacks[stack_id])


@metadata_bp.route('/metadata/target/<target_id>/<filter>')
def target_metadata(target_id: str, filter: str):
  return get_metadata(Target, context.targets[target_id], filter)


@metadata_bp.route('/metadata/mosaic/<mosaic_id>/<filter>')
def mosaic_metadata(mosaic_id: str, filter: str):
  return get_metadata(Mosaic, context.mosaics[mosaic_id], filter)
