from ray_infra import ray_local

import unittest
import ray
from context import context, get_cache
import numpy as np
from pathlib import Path

from testing import make_test_context, copy_testdata_into_cache

from artifacts.artifact import unpack_artifact
from mosaics.placement import place_small_image, place_large_image
from previews.types import QualityLevel
from metadata.image_metadata import load_metadata
from mosaics.viewbox import compute_mosaic_viewbox, Viewbox
from mosaics.types import TileIndices
from testing.test_stacks import make_composite_stack


class TestPlacement(unittest.TestCase):

  def setUp(self) -> None:
    self.input_dir_obj, self.output_dir_obj = make_test_context()

  def test_weird_wcs(self) -> None:
    cache = get_cache()
    copy_testdata_into_cache()

    ray.get(load_metadata.remote())

    print('### metadata.keys()', context.metadata.keys())
    print('### mosaics.keys()', context.mosaics.keys())

    mode = QualityLevel.MASTER
    mosaic_id = 'b1c2c11588a866ac'
    filter = 'H'
    tile_indices = TileIndices((0, 0))
    filename = 'testdata/Stacks/IC1318/masters/master_light_IC1318_RA_306.20_DEC_35.00_ZWO ASI6200MM Pro_2.21aspp_1.55deg_H_3bd18a01cedf3c6b.xisf'

    mosaic = context.mosaics[mosaic_id]
    stack = make_composite_stack(content_hash='stack_abc123')

    viewbox = compute_mosaic_viewbox(mode, mosaic)
    self.assertTrue(isinstance(viewbox, Viewbox))

    log, result = ray.get(
        place_small_image.remote(QualityLevel.SUBFRAME, mosaic, stack,
                                 tile_indices, filter, viewbox, filename))
    print('### result', result)
    self.assertTrue(isinstance(unpack_artifact(result), Path))

    log, result = ray.get(
        place_large_image.remote(QualityLevel.MASTER, mosaic, stack,
                                 tile_indices, filter, viewbox, filename))
    self.assertTrue(isinstance(unpack_artifact(result), Path))


if __name__ == "__main__":
  unittest.main()
