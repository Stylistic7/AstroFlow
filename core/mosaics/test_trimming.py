from ray_infra import ray_local

import matplotlib.pyplot as plt
import unittest
import shapely  # type: ignore
from shapely import Polygon  # type: ignore

from compute_graph import build_compute_graph
from context import context
from dependency_graph import run_graph

from mosaics.trimming import largest_interior_rectangle
from mosaics.orion_mosaic import add_orion_mosaic_images
from mosaics.debugging import draw_polygon, draw_spherical_polygon
from geometry.polygons import project_polygon
from testing.test_frames import make_synthetic_frames
from testing.test_context import make_test_context
from geometry.projections import mercator
from stacks.composite_stack import CompositeStack


class TestTrimming(unittest.TestCase):

  def setUp(self) -> None:
    self.input_dir_obj, self.output_dir_obj = make_test_context()

  def tearDown(self) -> None:
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_orion_mosaic(self) -> None:
    add_orion_mosaic_images(panes=[1, 2])

    run_graph(
        build_compute_graph(regroup=True,
                            make_previews=False,
                            run_main_processing_graph=False))

    stacks = context.stacks
    self.assertEqual(len(stacks), 2)

    targets = context.targets
    self.assertEqual(len(targets), 2)

    mosaics = context.mosaics
    self.assertEqual(len(mosaics), 1)
    mosaic = next(iter(mosaics.values()))

    projection = mercator.MercatorProjection()
    x, y = project_polygon(mosaic.footprint, projection)
    projected_footprint = Polygon(zip(x, y))
    print('###', list(projected_footprint.exterior.coords))

    plt.figure()

    union_poly = None
    for target in context.targets.values():
      draw_spherical_polygon(target.footprint, projection, color='blue')
      x, y = project_polygon(target.footprint, projection)
      poly = Polygon(zip(x, y))
      union_poly = poly if union_poly is None else union_poly.union(poly)

    print(f'### union_poly.exterior.coords {list(union_poly.exterior.coords)}')
    box = largest_interior_rectangle(union_poly)
    print(f'### box.exterior.coords {list(box.exterior.coords)}')
    x, y = zip(*box.exterior.coords)
    plt.plot(x, y, color='black')

    draw_spherical_polygon(mosaic.footprint, projection, color='red')

    plt.show()


if __name__ == "__main__":
  unittest.main()
