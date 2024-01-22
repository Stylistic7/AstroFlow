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


class TestDependencyGraph(unittest.TestCase):

  def test_simple_graph(self):
    receipt: Receipt = workflow.run(
        book_trip.bind("OAK", "SAN", ["6/12", "7/5"]))


if __name__ == "__main__":
  unittest.main()
