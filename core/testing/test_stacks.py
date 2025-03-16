import datetime
from typing import Any
from dataclasses import asdict

from spherical_geometry.polygon import SphericalPolygon  # type: ignore

from stacks.composite_stack import CompositeStack
from stacks.astrometric_attributes import AstrometricAttributes
from stacks.imaging_attributes import ImagingAttributes
from stacks.chronology_attributes import ChronologyAttributes


def make_composite_stack(**kwargs: Any) -> CompositeStack:
  chronology_attributes = asdict(
      ChronologyAttributes(
          oldest_date=datetime.datetime(2024, 2, 5, 16, 29, 43),
          newest_date=datetime.datetime(2024, 2, 5, 17, 29, 43),
      ))
  imaging_attributes = asdict(
      ImagingAttributes(
          image_type='light',
          camera='ZWO ASI6200MM Pro',
          filter='H',
          mean_temperature=-20.0,
          exposure_s=[300.0],
          integration_s=1500.0,
          width_pixels=9576,
          height_pixels=6488,
      ))
  astrometric_attributes = asdict(
      AstrometricAttributes(
          name='name',
          center=[50.0, 50.0],
          image_scale=1.0,
          rotation_angle=0.0,
          footprint=SphericalPolygon.from_cone(0.0, 90.0, 1.0),
          projected_footprints={},
      ))
  composite_stack_attributes = {
      'stable_hash': 'stable_hash',
      'content_hash': 'content_hash',
      'contents': [],
      'state': {}
  }

  attributes = (chronology_attributes | imaging_attributes |
                astrometric_attributes | composite_stack_attributes | kwargs)

  return CompositeStack(**attributes)
