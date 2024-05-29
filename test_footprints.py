import ray_local

import ray
import unittest
import shapely
from shapely import Polygon
import datetime

from context import context
from dependency_graph import run_graph, make_graph_from_list
from footprints import project_footprint
from make_test_frames import make_synthetic_frames
from mosaics import identification
from stack_identification import identify_stacks
from target_identification import group_stacks_into_targets
from spatial_association import find_components_by_cosine_similarity
from image_metadata import scatter_compute_image_metadata
from calibration_frames import identify_calibration_stacks, associate_with_stacks
from make_test_context import make_test_context
from mosaics.identification import identify_mosiacs
import mercator


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
    image_hashes, fits_headers, plate_solutions = make_synthetic_frames(
        number=3,
        prefix='foo',
        ra=180,
        dec=0,
        width_pixels=10800,
        height_pixels=7200,
        rotation_angle_deg=0,
        dither=0,
        dest_dir=context.input_dirs[0])

    # TODO: It's awkward that this has to be done both ways. Maybe split the
    # image_metadata gather op from the diskcache ops.
    context.image_hashes = image_hashes
    context.fits_headers = fits_headers
    context.plate_solutions = plate_solutions

    graph = _build_compute_graph()
    run_graph(graph)

    stacks = context.stacks
    self.assertEqual(len(stacks), 2)
    stack = next(iter(s for s in stacks.values() if s.geometric_hash))

    projected_footprint = project_footprint(stack.footprint_polygon(),
                                            self.proj)

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
    image_hashes, fits_headers, plate_solutions = make_synthetic_frames(
        number=3,
        prefix='foo',
        ra=180,
        dec=0,
        rotation_angle_deg=0,
        dither=0,
        image_scale=1.0,
        width_pixels=10800,
        height_pixels=7200,
        dest_dir=context.input_dirs[0])
    image_hashes, fits_headers, plate_solutions = make_synthetic_frames(
        number=5,
        prefix='bar',
        ra=180,
        dec=0,
        rotation_angle_deg=0,
        dither=0,
        image_scale=2.0,
        width_pixels=5400,
        height_pixels=3600,
        dest_dir=context.input_dirs[0],
        image_hashes=image_hashes,
        fits_headers=fits_headers,
        plate_solutions=plate_solutions)

    context.image_hashes = image_hashes
    context.fits_headers = fits_headers
    context.plate_solutions = plate_solutions

    graph = _build_compute_graph()
    run_graph(graph)

    stacks = context.stacks
    self.assertEqual(len(stacks), 4)

    targets = context.targets
    self.assertEqual(len(targets), 1)

    target = next(iter(targets.values()))

    projected_footprint = project_footprint(target.footprint_polygon(),
                                            self.proj)

    self.assertTrue(
        shapely.equals(
            projected_footprint,
            Polygon([(3.1677617688622615, -0.017441585851835084),
                     (3.1154235383173243, -0.017441585851835084),
                     (3.1154235383173243, 0.017441585851835202),
                     (3.1677617688622615, 0.017441585851835202),
                     (3.1677617688622615, -0.017441585851835084)])))

  def test_mosaic(self):
    # Two stacks that overlap by a degree in RA.
    image_hashes, fits_headers, plate_solutions = make_synthetic_frames(
        number=3,
        prefix='foo',
        ra=180,
        dec=0,
        rotation_angle_deg=0,
        dither=0,
        image_scale=1.0,
        width_pixels=10800,
        height_pixels=7200,
        dest_dir=context.input_dirs[0])
    image_hashes, fits_headers, plate_solutions = make_synthetic_frames(
        number=5,
        prefix='bar',
        ra=182,
        dec=0,
        rotation_angle_deg=0,
        dither=0,
        image_scale=1.0,
        width_pixels=10800,
        height_pixels=7200,
        dest_dir=context.input_dirs[0],
        image_hashes=image_hashes,
        fits_headers=fits_headers,
        plate_solutions=plate_solutions)

    context.image_hashes = image_hashes
    context.fits_headers = fits_headers
    context.plate_solutions = plate_solutions

    graph = _build_compute_graph()
    run_graph(graph)

    stacks = context.stacks
    self.assertEqual(len(stacks), 4)

    targets = context.targets
    self.assertEqual(len(targets), 2)

    mosaics = context.mosaics
    self.assertEqual(len(mosaics), 1)
    mosaic = next(iter(mosaics.values()))

    projected_footprint = project_footprint(mosaic.footprint_polygon(),
                                            self.proj)

    self.assertTrue(
        shapely.equals(
            projected_footprint,
            Polygon([(3.1154211159087506, -0.01744400761566263),
                     (3.1154211159087506, 0.017444007615662508),
                     (3.159045946109729, 0.01744732539865246),
                     (3.202670776310722, 0.017444007615662508),
                     (3.202670776310722, -0.01744400761566263),
                     (3.1590459461097176, -0.017447325398652496),
                     (3.1154211159087506, -0.01744400761566263)])))


if __name__ == "__main__":
  unittest.main()
