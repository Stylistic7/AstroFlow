import ray_local

import ray
import unittest
import shapely
from shapely import Polygon
import datetime

import ray_diskcache
from dependency_graph import run_graph, make_graph_from_list
from context import Context
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

default_context = dict(cosine_similarity_threshold=0.99999,
                       min_component_members=3,
                       max_run_gap=datetime.timedelta(minutes=15),
                       min_frames_in_run=3,
                       min_frames_in_stack=3,
                       quarantined_files=ray.put([]))


def _build_compute_graph():
  return make_graph_from_list([(scatter_compute_image_metadata, 'all'),
                               find_components_by_cosine_similarity,
                               identify_stacks, identify_calibration_stacks,
                               associate_with_stacks, group_stacks_into_targets,
                               identify_mosiacs])


class TestFootprints(unittest.TestCase):

  def setUp(self):
    self.context, self.input_dir_obj, self.output_dir_obj = make_test_context()
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
        dest_dir=self.context.input_dirs[0])

    # TODO: It's awkward that this has to be done both ways. Maybe split the
    # image_metadata gather op from the diskcache ops.
    self.context.image_hashes = ray.put(image_hashes)
    self.context.fits_headers = ray.put(fits_headers)
    self.context.plate_solutions = ray.put(plate_solutions)
    ray_diskcache.set('image_hashes', image_hashes)
    ray_diskcache.set('fits_headers', fits_headers)
    ray_diskcache.set('plate_solutions', plate_solutions)

    graph = _build_compute_graph()
    self.context, _ = run_graph(graph, self.context)

    stacks = ray.get(self.context.stacks)
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
        dest_dir=self.context.input_dirs[0])
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
        dest_dir=self.context.input_dirs[0],
        image_hashes=image_hashes,
        fits_headers=fits_headers,
        plate_solutions=plate_solutions)

    self.context.image_hashes = ray.put(image_hashes)
    self.context.fits_headers = ray.put(fits_headers)
    self.context.plate_solutions = ray.put(plate_solutions)
    ray_diskcache.set('image_hashes', image_hashes)
    ray_diskcache.set('fits_headers', fits_headers)
    ray_diskcache.set('plate_solutions', plate_solutions)

    graph = _build_compute_graph()
    self.context, _ = run_graph(graph, self.context)

    stacks = ray.get(self.context.stacks)
    self.assertEqual(len(stacks), 4)

    targets = ray.get(self.context.targets)
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
        dest_dir=self.context.input_dirs[0])
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
        dest_dir=self.context.input_dirs[0],
        image_hashes=image_hashes,
        fits_headers=fits_headers,
        plate_solutions=plate_solutions)

    self.context.image_hashes = ray.put(image_hashes)
    self.context.fits_headers = ray.put(fits_headers)
    self.context.plate_solutions = ray.put(plate_solutions)
    ray_diskcache.set('image_hashes', image_hashes)
    ray_diskcache.set('fits_headers', fits_headers)
    ray_diskcache.set('plate_solutions', plate_solutions)

    graph = _build_compute_graph()
    self.context, _ = run_graph(graph, self.context)

    stacks = ray.get(self.context.stacks)
    self.assertEqual(len(stacks), 4)

    targets = ray.get(self.context.targets)
    self.assertEqual(len(targets), 2)

    mosaics = ray.get(self.context.mosaics)
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
