import ray_local

import ray
import unittest
import shapely
from shapely import Polygon
import datetime

from context import Context
from footprints import project_footprint
from make_test_frames import make_synthetic_frames
from mosaics import identification
from stack_identification import identify_stacks
from calibration_frames import associate_calibration_frames
from target_identification import group_stacks_into_targets
from spatial_association import find_components_by_cosine_similarity
import mercator

default_context = dict(cosine_similarity_threshold=0.99999,
                       min_component_members=3,
                       max_run_gap=datetime.timedelta(minutes=15),
                       min_frames_in_run=3,
                       min_frames_in_stack=3,
                       quarantined_files=ray.put([]))


class TestFootprints(unittest.TestCase):

  def setUp(self):
    self.proj = mercator.MercatorProjection()

  def test_stack(self):
    image_hashes, fits_headers, plate_solutions = make_synthetic_frames(
        number=10,
        prefix='foo',
        ra=180,
        dec=0,
        width_pixels=10800,
        height_pixels=7200,
        rotation_angle_deg=0,
        dither=0)

    context = Context(
        image_hashes=ray.put(image_hashes),
        fits_headers=ray.put(fits_headers),
        plate_solutions=ray.put(plate_solutions)) | default_context

    context, _ = ray.get(find_components_by_cosine_similarity.remote(context))
    context, _ = ray.get(identify_stacks.remote(context))

    stacks = ray.get(context.stacks)
    self.assertEqual(len(stacks), 1)
    stack = next(iter(stacks))

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
        number=10,
        prefix='foo',
        ra=180,
        dec=0,
        rotation_angle_deg=0,
        dither=0,
        image_scale=1.0,
        width_pixels=10800,
        height_pixels=7200)
    image_hashes, fits_headers, plate_solutions = make_synthetic_frames(
        number=10,
        prefix='bar',
        ra=180,
        dec=0,
        rotation_angle_deg=0,
        dither=0,
        image_scale=2.0,
        width_pixels=5400,
        height_pixels=3600,
        image_hashes=image_hashes,
        fits_headers=fits_headers,
        plate_solutions=plate_solutions)

    context = Context(
        image_hashes=ray.put(image_hashes),
        fits_headers=ray.put(fits_headers),
        plate_solutions=ray.put(plate_solutions)) | default_context

    context, _ = ray.get(find_components_by_cosine_similarity.remote(context))
    context, _ = ray.get(identify_stacks.remote(context))
    context, _ = ray.get(associate_calibration_frames.remote(context))
    context, _ = ray.get(group_stacks_into_targets.remote(context))

    stacks = ray.get(context.stacks)
    self.assertEqual(len(stacks), 2)

    targets = ray.get(context.targets)
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
        number=10,
        prefix='foo',
        ra=180,
        dec=0,
        rotation_angle_deg=0,
        dither=0,
        image_scale=1.0,
        width_pixels=10800,
        height_pixels=7200)
    image_hashes, fits_headers, plate_solutions = make_synthetic_frames(
        number=10,
        prefix='bar',
        ra=182,
        dec=0,
        rotation_angle_deg=0,
        dither=0,
        image_scale=1.0,
        width_pixels=10800,
        height_pixels=7200,
        image_hashes=image_hashes,
        fits_headers=fits_headers,
        plate_solutions=plate_solutions)

    context = Context(
        image_hashes=ray.put(image_hashes),
        fits_headers=ray.put(fits_headers),
        plate_solutions=ray.put(plate_solutions)) | default_context

    context, _ = ray.get(find_components_by_cosine_similarity.remote(context))
    context, _ = ray.get(identify_stacks.remote(context))
    context, _ = ray.get(associate_calibration_frames.remote(context))
    context, _ = ray.get(group_stacks_into_targets.remote(context))

    stacks = ray.get(context.stacks)
    self.assertEqual(len(stacks), 2)

    targets = ray.get(context.targets)
    self.assertEqual(len(targets), 2)

    context, _ = ray.get(identification.identify_mosiacs.remote(context))

    mosaics = ray.get(context.mosaics)
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
