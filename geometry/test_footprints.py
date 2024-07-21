import ray_local

import ray
import unittest
import shapely
from shapely import Polygon
import datetime

from context import context
from dependency_graph import run_graph, make_graph_from_list
from geometry.footprints import project_footprint
from testing.test_frames import make_synthetic_frames
from mosaics import identification
from stacks.identification import identify_stacks
from target_identification import group_stacks_into_targets
from spatial_association import find_components_by_cosine_similarity
from metadata.image_metadata import scatter_compute_image_metadata
from calibration_frames import identify_calibration_stacks, associate_with_stacks
from testing.test_context import make_test_context
from mosaics.identification import identify_mosiacs
from geometry.projections import mercator


def _build_compute_graph():
  return make_graph_from_list([(scatter_compute_image_metadata, 'all'),
                               find_components_by_cosine_similarity,
                               identify_stacks, identify_calibration_stacks,
                               associate_with_stacks, group_stacks_into_targets,
                               identify_mosiacs])


class TestFootprints(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()
    self.proj = mercator.MercatorProjection()

  def test_stack(self):
    make_synthetic_frames(number=3,
                          prefix='foo',
                          ra=180,
                          dec=0,
                          width_pixels=10800,
                          height_pixels=7200,
                          rotation_angle_deg=0,
                          dither=0,
                          dest_dir=context.input_dirs[0])

    graph = _build_compute_graph()
    run_graph(graph)

    stacks = context.stacks
    self.assertEqual(len(stacks), 2)
    stack = next(iter(s for s in stacks.values() if s.stable_hash))

    projected_footprint = project_footprint(stack.footprint, self.proj)

    self.assertTrue(
        shapely.equals(
            projected_footprint,
            Polygon([(3.1154211159087506, -0.01744400761566263),
                     (3.1154211159087506, 0.017444007615662508),
                     (3.167764191270835, 0.017444007615662508),
                     (3.167764191270835, -0.01744400761566263),
                     (3.1154211159087506, -0.01744400761566263)])))

  def test_target(self):
    # Two stacks with the same footprint, but different image scales. They
    # should be merged into a single target.
    make_synthetic_frames(number=3,
                          prefix='foo',
                          ra=180,
                          dec=0,
                          rotation_angle_deg=0,
                          dither=0,
                          image_scale=1.0,
                          width_pixels=10800,
                          height_pixels=7200,
                          dest_dir=context.input_dirs[0])
    make_synthetic_frames(number=5,
                          prefix='bar',
                          ra=180,
                          dec=0,
                          rotation_angle_deg=0,
                          dither=0,
                          image_scale=2.0,
                          width_pixels=5400,
                          height_pixels=3600,
                          dest_dir=context.input_dirs[0])

    graph = _build_compute_graph()
    run_graph(graph)

    stacks = context.stacks
    self.assertEqual(len(stacks), 4)

    targets = context.targets
    self.assertEqual(len(targets), 1)

    target = next(iter(targets.values()))

    projected_footprint = project_footprint(target.footprint, self.proj)
    print('### projected_footprint', projected_footprint)

    self.assertTrue(
        shapely.equals_exact(projected_footprint,
                             Polygon([(3.1154211159087506, -0.0174440076156626),
                                      (3.1154211159087506, 0.0174440076156625),
                                      (3.167764191270835, 0.0174440076156625),
                                      (3.167764191270835, -0.0174440076156626),
                                      (3.1154211159087506, -0.0174440076156626)
                                     ]),
                             tolerance=1e-5))

  def test_mosaic(self):
    # Two stacks that overlap by a degree in RA.
    make_synthetic_frames(number=3,
                          prefix='foo',
                          ra=180,
                          dec=0,
                          rotation_angle_deg=0,
                          dither=0,
                          image_scale=1.0,
                          width_pixels=10800,
                          height_pixels=7200,
                          dest_dir=context.input_dirs[0])
    make_synthetic_frames(number=5,
                          prefix='bar',
                          ra=182,
                          dec=0,
                          rotation_angle_deg=0,
                          dither=0,
                          image_scale=1.0,
                          width_pixels=10800,
                          height_pixels=7200,
                          dest_dir=context.input_dirs[0])

    graph = _build_compute_graph()
    run_graph(graph)

    stacks = context.stacks
    self.assertEqual(len(stacks), 4)

    targets = context.targets
    self.assertEqual(len(targets), 2)

    mosaics = context.mosaics
    self.assertEqual(len(mosaics), 1)
    mosaic = next(iter(mosaics.values()))

    projected_footprint = project_footprint(mosaic.footprint, self.proj)

    print('###', list(projected_footprint.exterior.coords))
    self.assertTrue(
        shapely.equals_exact(projected_footprint,
                             Polygon([
                                 (3.150327700948637, -0.01744400761566263),
                                 (3.167764191270835, 0.017444007615662508),
                                 (3.202670776310722, 0.017444007615662508),
                                 (3.202670776310722, -0.01744400761566263),
                                 (3.1154211159087506, -0.01744400761566263),
                                 (3.1154211159087506, 0.017444007615662508),
                                 (3.150327700948637, 0.017444007615662508),
                                 (3.150327700948637, -0.01744400761566263)
                             ]),
                             tolerance=1e-5))


if __name__ == "__main__":
  unittest.main()
