from ray_infra import ray_local

import ray
import numpy as np
import unittest
from pathlib import Path

from testing import make_test_context
from context import context
from products import register_image_producer
from file_io.types import TiffFilePath
from file_io.tiff_file_io import write_tiff_image
from task.astroflow_node import task


@task.remote
@register_image_producer(object_type="stack", level="subframe", tiled=False)
def dummy_generator(stack_id: str) -> Path:
  output_path = Path(context.output_dir) / f"{stack_id}.png"
  image = np.zeros((100, 100))
  write_tiff_image(output_path, image)

  return TiffFilePath(output_path)


class TestRegisterDecorator(unittest.TestCase):

  def setUp(self) -> None:
    # Start with an empty context.
    self.input_dir_obj, self.output_dir_obj = make_test_context()

  def tearDown(self) -> None:
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_register_decorator(self) -> None:
    # Call the dummy generator
    image_path = ray.get(dummy_generator.remote("abc123"))

    assert image_path.exists()

    # Check thumbnail
    thumb_key = ("thumbnail", "stack", "subframe", "abc123")
    assert thumb_key in context
    thumb_data = context[thumb_key]
    assert isinstance(thumb_data, bytes)

    # Check metadata
    meta_key = ("metadata", "stack", "abc123")
    assert meta_key in context
    metadata = context[meta_key]
    assert "subframe" in metadata

    subframe_meta = metadata["subframe"]
    assert subframe_meta["type"] == "png"
    assert subframe_meta["image_url"] == "/image/stack/subframe/abc123"
    assert subframe_meta["thumbnail_url"] == "/thumbnail/stack/subframe/abc123"
    assert isinstance(subframe_meta["created_at"], float)
    # assert subframe_meta["size"] == (100, 100)
