import io
import os
import tempfile
import subprocess
import time
from pathlib import Path
from xisf import XISF
import plate_solving
import config
import astropy.io.fits as pyfits

from cache import cache
import pixinsight_shell_script
import pixinsight_image_solver
import pixinsight_mosaic_by_coordinates
import trim_mosaic_tile
import spatial_association


if __name__ == '__main__':
  # filenames = [
  #     '/Users/nobody/Downloads/Veil Pano/master_light_Filamentary_nebula_RA_311_5_DEC_30_2_C11_H_v1_post_registered.xisf',
  #     '/Users/nobody/Downloads/Veil Pano/master_light_Filamentary_nebula_RA_313_1_DEC_30_0_C11_H_v1_post_registered.xisf'
  # ]

  # filenames = pixinsight_image_solver.run(filenames)
  # filenames = pixinsight_mosaic_by_coordinates.run(filenames)

  # for filename in filenames:
  #   filenames = trim_mosaic_tiles(filename)

