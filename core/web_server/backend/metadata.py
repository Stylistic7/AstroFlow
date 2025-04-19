import os
from flask import Blueprint, jsonify

from context import context
from stacks.composite_stack import CompositeStack
from targets.types import Target
from mosaics.types import Mosaic
from mosaics.viewbox import compute_mosaic_viewbox
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

    if object_type == Mosaic:
      mosaic = args[0]  # FIXME
      viewbox = compute_mosaic_viewbox(level, mosaic)
      polygon = project_to_viewbox(mosaic.footprint, viewbox)
      coords = list(polygon.exterior.coords)
      result['projected_footprint'] = coords

    return jsonify(result)


@metadata_bp.route('/metadata/stack/<stack_id>')
def stack_metadata(stack_id: str):
  return get_metadata(CompositeStack, context.stacks[stack_id])


@metadata_bp.route('/metadata/target/<target_id>')
def target_metadata(target_id: str):
  return get_metadata(Target, context.targets[target_id])


@metadata_bp.route('/metadata/mosaic/<mosaic_id>/<filter>')
def mosaic_metadata(mosaic_id: str, filter: str):
  return get_metadata(Mosaic, context.mosaics[mosaic_id], filter)
