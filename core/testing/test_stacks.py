import datetime
from typing import Any
from dataclasses import asdict

from spherical_geometry.polygon import SingleSphericalPolygon  # type: ignore

from stacks.composite_stack import CompositeStack
from attributes import AstrometricAttributes, ChronologyAttributes, ImagingAttributes
from attributes.astrometric_attributes import healpix_cells_from_poly

def make_composite_stack(**kwargs: Any) -> CompositeStack:
  chronology_attributes = asdict(
      ChronologyAttributes(
          stable_hash='stable_hash',
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
  footprint = SingleSphericalPolygon.from_radec([98.01, 102.01, 102.01, 98.01],
                                               [-1, -1, 1, 1],
                                               center=[100.01, 0])
  healpix_cells = healpix_cells_from_poly(footprint)
  astrometric_attributes = asdict(
      AstrometricAttributes(
          name='name',
          center=[50.0, 50.0],
          image_scale=1.0,
          rotation_angle=0.0,
          footprint=footprint,
          healpix_cells=healpix_cells,
          projected_footprints={},
      ))
  composite_stack_attributes = {'contents': [], 'state': {}}

  attributes = (chronology_attributes | imaging_attributes |
                astrometric_attributes | composite_stack_attributes | kwargs)

  return CompositeStack(**attributes)
