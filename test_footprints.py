import unittest
import numpy as np
import shapely
from shapely import Polygon

from cache import cache
from stack import Stack
from target import Target
from mosaic import Mosaic
from footprints import project_footprint
from make_test_frames import make_test_frames
import step_mosaic
from mosaics import identification
from stack_identification import identify_stacks
from astro_flow_direct import group_stacks_into_targets
from spatial_association import find_components_by_cosine_similarity
import mercator


class TestFootprints(unittest.TestCase):
  def setUp(self):
    self.proj = mercator.MercatorProjection()

  def test_stack(self):
    image_hashes, fits_headers, plate_solutions = make_test_frames(
        number=10,
        prefix='foo',
        ra=180,
        dec=0,
        width_pixels=10800,
        height_pixels=7200,
        rotation_angle_deg=0,
        dither=0)

    spatial_components = find_components_by_cosine_similarity(
        image_hashes.keys(), plate_solutions, fits_headers)
    stacks = identify_stacks(spatial_components, plate_solutions, fits_headers)
    stacks = {s.id: s for s in stacks}

    self.assertEqual(len(stacks), 1)
    stack = next(iter(stacks.values()))

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
    image_hashes, fits_headers, plate_solutions = make_test_frames(
        number=10,
        prefix='foo',
        ra=180,
        dec=0,
        rotation_angle_deg=0,
        dither=0,
        image_scale=1.0,
        width_pixels=10800,
        height_pixels=7200)
    image_hashes, fits_headers, plate_solutions = make_test_frames(
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

    spatial_components = find_components_by_cosine_similarity(
        image_hashes.keys(), plate_solutions, fits_headers)
    stacks = identify_stacks(spatial_components, plate_solutions, fits_headers)
    stacks = {s.id: s for s in stacks}

    self.assertEqual(len(stacks), 2)

    targets = group_stacks_into_targets(stacks, plate_solutions, fits_headers)
    self.assertEqual(len(targets), 1)
    target = next(iter(targets.values()))

    projected_footprint = project_footprint(target.footprint, self.proj)

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
    image_hashes, fits_headers, plate_solutions = make_test_frames(
        number=10,
        prefix='foo',
        ra=180,
        dec=0,
        rotation_angle_deg=0,
        dither=0,
        image_scale=1.0,
        width_pixels=10800,
        height_pixels=7200)
    image_hashes, fits_headers, plate_solutions = make_test_frames(
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

    spatial_components = find_components_by_cosine_similarity(
        image_hashes.keys(), plate_solutions, fits_headers)
    stacks = identify_stacks(spatial_components, plate_solutions, fits_headers)
    stacks = {s.id: s for s in stacks}

    self.assertEqual(len(stacks), 2)

    targets = group_stacks_into_targets(stacks, plate_solutions, fits_headers)
    self.assertEqual(len(targets), 2)

    mosaics = identification.identify_mosiacs(targets)
    self.assertEqual(len(mosaics), 1)
    mosaic = next(iter(mosaics))

    projected_footprint = project_footprint(mosaic.footprint, self.proj)

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