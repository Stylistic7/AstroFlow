import matplotlib.pyplot as plt
from shapely import Polygon  # type: ignore

from compute_graph import build_compute_graph
from context import context
from dependency_graph import run_graph
from mosaics.trimming import largest_interior_rectangle
from mosaics.orion_mosaic import add_orion_mosaic_images
from mosaics.debugging import draw_spherical_polygon
from geometry.polygons import project_polygon
from testing.test_context import make_test_context
from geometry.projections import mercator


def test_orion_mosaic(test_context) -> None:
  add_orion_mosaic_images()

  run_graph(build_compute_graph())

  stacks = context.stacks
  assert len(stacks) == 4

  targets = context.targets
  assert len(targets) == 4

  mosaics = context.mosaics
  assert len(mosaics) == 1
  mosaic = next(iter(mosaics.values()))

  projection = mercator.MercatorProjection()
  x, y = project_polygon(mosaic.footprint, projection)
  projected_footprint = Polygon(zip(x, y))

  # TODO: Write a real test with an assertion.

  # plt.figure()

  # union_poly = None
  # for target in context.targets.values():
  #   draw_spherical_polygon(target.footprint, projection, color='blue')
  #   x, y = project_polygon(target.footprint, projection)
  #   poly = Polygon(zip(x, y))
  #   union_poly = poly if union_poly is None else union_poly.union(poly)

  # print(f'### union_poly.exterior.coords {list(union_poly.exterior.coords)}')
  # box = largest_interior_rectangle(union_poly, epsilon=0)
  # print(f'### box.exterior.coords {list(box.exterior.coords)}')
  # x, y = zip(*box.exterior.coords)
  # plt.plot(x, y, color='black')

  # draw_spherical_polygon(mosaic.footprint, projection, color='red')

  # plt.show()
