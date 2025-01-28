from ray_infra import ray_local

import unittest
import ray
from context import context, get_cache
import numpy as np
from pathlib import Path

from testing import make_test_context, copy_testdata_into_cache

from artifacts.artifact import unpack_artifact
from mosaics.placement import place_image
from previews.types import QualityLevel
from metadata.image_metadata import load_metadata
from mosaics.viewbox import compute_mosaic_viewbox, Viewbox
from mosaics.types import TileIndices


class TestPlacement(unittest.TestCase):

  def setUp(self) -> None:
    self.input_dir_obj, self.output_dir_obj = make_test_context()

  def test_weird_wcs(self) -> None:
    cache = get_cache()
    copy_testdata_into_cache()

    ray.get(load_metadata.remote())

    print('### metadata.keys()', context.metadata.keys())
    print('### mosaics.keys()', context.mosaics.keys())

    mosaic_id = 'b1c2c11588a866ac'
    filter = 'H'
    stack_id = '2c71f4db63b27773'
    content_hash = '187b447aad352e0f'
    tile_indices = TileIndices((0, 0))
    filename = 'testdata/Stacks/IC1318/masters/master_light_IC1318_RA_306.20_DEC_35.00_ZWO ASI6200MM Pro_2.21aspp_1.55deg_H_187b447aad352e0f.xisf'

    mosaic = context.mosaics[mosaic_id]

    viewbox = compute_mosaic_viewbox(mosaic)
    self.assertTrue(isinstance(viewbox, Viewbox))

    log, result = ray.get(
        place_image.remote(QualityLevel.SUBFRAME, mosaic, tile_indices, viewbox,
                           filename))
    print('### result', result)
    self.assertTrue(isinstance(unpack_artifact(result), np.ndarray))

    log, result = ray.get(
        place_image.remote(QualityLevel.MASTER, mosaic, tile_indices, viewbox,
                           filename))
    self.assertTrue(isinstance(unpack_artifact(result), Path))


if __name__ == "__main__":
  unittest.main()
