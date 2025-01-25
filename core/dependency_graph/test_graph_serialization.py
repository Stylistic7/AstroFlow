from ray_infra import ray_local

import unittest
import ray

from context import get_cache
from context import context
from compute_graph.compute_graph import load_metadata
from dependency_graph.serialization import pickle_graph
from testing.testdata_cache import copy_testdata_into_cache
from mosaics import identification
from mosaics.compute_graph import build_mosaic_subgraph
from previews import QualityLevel


class TestGraphSerialization(unittest.TestCase):

  def test_big_cygnus_pano(self) -> None:
    cache = get_cache()
    copy_testdata_into_cache()

    ray.get(load_metadata.remote())

    ray.get(identification.identify_mosaics.remote(debug=False))
    stacks = context.stacks
    targets = context.targets
    mosaics = context.mosaics

    self.assertEqual(len(mosaics), 1)
    mosaic = next(iter(mosaics.values()))
    print('### mosaic', mosaic)

    target_ids = mosaic.target_ids
    print('### target_ids', target_ids)

    _, graph = ray.get(
        build_mosaic_subgraph.remote(QualityLevel.SUBFRAME, mosaic))

    serialized = pickle_graph(graph)


if __name__ == '__main__':
  unittest.main()
