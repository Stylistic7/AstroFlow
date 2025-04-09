import io
import os
import time
import threading
import large_image
from pathlib import Path
import numpy as np
import tifffile
from flask import Blueprint, send_file, abort, jsonify

from context import context
from stacks.artifacts import output_filename as stack_output_filename
from mosaics.artifacts import tile_filename as mosaic_output_filename
from file_io.generic import read_image
from file_io.types import TiffFilePath
from previews import autostretch
from stacks.composite_stack import CompositeStack
from targets.types import Target
from mosaics.types import Mosaic
from products.registry import get_image_producer
from previews import QualityLevel

objects_bp = Blueprint("objects", __name__)


@objects_bp.route('/metadata/<object_type>/<object_id>')
def metadata(object_type, object_id):
  class_for_type = {'stack': CompositeStack, 'target': Target, 'mosaic': Mosaic}

  if object_type == 'stack':
    object = context.stacks[object_id]
  elif object_type == 'target':
    object = context.targets[object_id]
  elif object_type == 'mosaic':
    object = context.mosaics[object_id]

  result = {'levels': {}}
  for level in [
      QualityLevel.SUBFRAME, QualityLevel.MASTER, QualityLevel.POST_PROCESSED
  ]:
    fn = get_image_producer(class_for_type[object_type], level)
    if fn is None:
      continue

    filename = context.output_dir / fn.filename_fn(object)
    print(f'### quality_level {level} filename {filename}')

    result['levels'][level.name.lower()] = {'image_path': os.fspath(filename)}

  return jsonify(result)


@objects_bp.route("/image/<object_type>/<object_id>")
def image(object_type, object_id, z, x, y):
  return abort(404)
