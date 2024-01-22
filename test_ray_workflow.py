# yapf: disable
import os
os.environ["ASTROFLOW_MOCK_CACHE"] = "true"

from cache import cache

# TODO: Need to use a temp directory for ray_storage so the test is hermetic.
import ray
from ray import workflow
ray.init(
    address='local',
    namespace='AstroFlow',
    resources={"diskcache": 1},
    ignore_reinit_error=True,
    runtime_env={"env_vars": {
        "ASTROFLOW_MOCK_CACHE": "true"
    }},
    storage="/tmp/ray_storage",
)
# yapf: enable

import unittest
from typing import List
import tempfile
import pathlib

import ray_diskcache
from make_test_frames import make_test_frames
from image_metadata import compute_image_metadata_ray, store_image_metadata_ray
from spatial_association import find_components_by_cosine_similarity_ray
from stack_identification import identify_stacks_ray
from calibration_frames import associate_calibration_frames
from image_handling import scatter_stack_thumbnails


class Flight(dict):
  pass


class Hotel(dict):
  pass


class Receipt(dict):
  pass


@ray.remote
def book_flight(origin: str, dest: str, date: str) -> Flight:
  return Flight()


@ray.remote
def book_hotel(origin: str, dates: List[str]) -> Hotel:
  return Hotel()


@ray.remote
def finalize_or_cancel(flights: List[Flight], hotels: List[Hotel]) -> Receipt:
  return Receipt()


@ray.remote
def book_trip(origin: str, dest: str, dates: List[str]) -> Receipt:
  # Note that the workflow engine will not begin executing
  # child workflows until the parent task returns.
  # This avoids task overlap and ensures recoverability.
  f1 = book_flight.bind(origin, dest, dates[0])
  f2 = book_flight.bind(dest, origin, dates[1])
  hotel = book_hotel.bind(dest, dates)
  return workflow.continuation(finalize_or_cancel.bind([f1, f2], [hotel]))


@ray.remote
def astroflow_compute_graph(new_filenames):
  compute_metadata = [compute_image_metadata_ray.bind(f) for f in new_filenames]
  context = store_image_metadata_ray.bind(compute_metadata)
  context = find_components_by_cosine_similarity_ray.bind(context)
  context = identify_stacks_ray.bind(context)
  context = associate_calibration_frames.bind(context)
  context = scatter_stack_thumbnails.bind(context)

  return workflow.continuation(context)


class TestDependencyGraph(unittest.TestCase):

  def test_simple_graph(self):
    receipt: Receipt = workflow.run(
        book_trip.bind("OAK", "SAN", ["6/12", "7/5"]))

  def test_astroflow_graph(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      make_test_frames(number=3,
                       project_dir=tmp_dir,
                       testdata_image='bubble_nebula')
      make_test_frames(number=5,
                       project_dir=tmp_dir,
                       testdata_image='sadr_region')
      make_test_frames(number=7, image_type='Dark', project_dir=tmp_dir)
      make_test_frames(number=11, image_type='Bias', project_dir=tmp_dir)
      make_test_frames(number=13, image_type='Flat', project_dir=tmp_dir)

      filenames = sorted([
          str(path.absolute()) for path in pathlib.Path(tmp_dir).glob("*.fit*")
      ])
      self.assertEqual(len(filenames), 39)

      image_hashes = ray_diskcache.get('image_hashes', {})
      new_filenames = list(set(filenames) - set(image_hashes.values()))
      print(f'### Importing {len(new_filenames)} files')

      result = workflow.run(astroflow_compute_graph.bind(new_filenames),
                            workflow_id='AstroFlow')
      print('### result', result)


if __name__ == "__main__":
  unittest.main()
