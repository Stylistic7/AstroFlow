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
import diskcache

import pixinsight_shell_script
import pixinsight_image_solver
import pixinsight_mosaic_by_coordinates

# cache = diskcache.Cache(os.path.join(os.getcwd(), '.cache'))

# shell_script = """
# run -x -p="solver_catalogMode,3" -p="solver_catalog,GaiaDR3_XPSD" \
#   -p="solver_useActive,false" -p="solver_files,{filenames}" \
#   -p="non_interactive,true" -p="isGlobalTarget,true" -p="isViewTarget,false" \
#   "/Applications/PixInsight/src/scripts/AdP/ImageSolver.js"

# run -x -p="engine_files,/Users/nobody/Downloads/Veil Pano/master_light_Filamentary_nebula_RA_311_5_DEC_30_2_C11_H_v1_post_registered_WCS.xisf|/Users/nobody/Downloads/Veil Pano/master_light_Filamentary_nebula_RA_313_1_DEC_30_0_C11_H_v1_post_registered_WCS.xisf" \
#   -p="engine_centerCoordsAuto,true" -p="engine_resolutionAuto,true" \
#   -p="engine_rotationAuto,true" -p="engine_projectionAuto,true" \
#   -p="engine_dimensionsAuto,true" -p="engine_qualityHQ,true" \
#   -p="engine_outputDir,/Users/nobody/Downloads/Veil Pano" \
#   -p="non_interactive,true" \
#   "/Applications/PixInsight/src/scripts/AdP/MosaicByCoordinates.js"

# ![echo 'done' > {sentinel_filename}
# """

if __name__ == '__main__':
  filenames = [
      '/Users/nobody/Downloads/Veil Pano/master_light_Filamentary_nebula_RA_311_5_DEC_30_2_C11_H_v1_post_registered.xisf',
      '/Users/nobody/Downloads/Veil Pano/master_light_Filamentary_nebula_RA_313_1_DEC_30_0_C11_H_v1_post_registered.xisf'
  ]

  pixinsight_image_solver.run(filenames)
  pixinsight_mosaic_by_coordinates.run(filenames)