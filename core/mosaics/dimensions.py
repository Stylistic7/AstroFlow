import numpy as np

from . import classes
from . import projection


def get_mosaic_dimensions(panels):
  min_ra, min_dec, ra_span, dec_span = projection.compute_viewbox(
      [p.plate_solution for p in panels])
  print(f'### min_ra {min_ra} ra_span {ra_span} '
        f'min_dec {min_dec} dec_span {dec_span}')

  # Adjust the height by the cos of the dec, so the resulting mosaic won't be
  # skinny.
  #
  # TODO: Implement more principled forms of correction, and choices of
  # projection.
  #
  # TODO: Select output resolution more intelligently, maybe by megapixel?
  # dec_resolution = 0.0002
  dec_resolution = dec_span / 500
  ra_resolution = dec_resolution / np.cos(np.deg2rad(min_dec + dec_span / 2.0))

  height, width = int(np.ceil(ra_span / ra_resolution)), int(
      np.ceil(dec_span / dec_resolution))
  print(f'### Final mosaic will have width {width} height {height}')

  return classes.MosaicDimensions(width=width,
                                  height=height,
                                  min_ra=min_ra,
                                  ra_span=ra_span,
                                  min_dec=min_dec,
                                  dec_span=dec_span,
                                  ra_resolution=ra_resolution,
                                  dec_resolution=dec_resolution)
