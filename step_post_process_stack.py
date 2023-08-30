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

cache = diskcache.Cache(os.path.join(os.getcwd(), '.cache'))

post_process_script = """
var P_bxt = new BlurXTerminator;
P_bxt.ai_file = "/Applications/PixInsight/library/BlurXTerminator.2.mlpackage";
P_bxt.correct_only = false;
P_bxt.correct_first = false;
P_bxt.nonstellar_then_stellar = false;
P_bxt.lum_only = false;
P_bxt.sharpen_stars = 0.25;
P_bxt.adjust_halos = 0.00;
P_bxt.nonstellar_psf_diameter = 0.00;
P_bxt.auto_nonstellar_psf = true;
P_bxt.sharpen_nonstellar = 0.90;

var P_abe = new AutomaticBackgroundExtractor;
P_abe.tolerance = 1.000;
P_abe.deviation = 0.800;
P_abe.unbalance = 1.800;
P_abe.minBoxFraction = 0.050;
P_abe.maxBackground = 1.0000;
P_abe.minBackground = 0.0000;
P_abe.useBrightnessLimits = false;
P_abe.polyDegree = 4;
P_abe.boxSize = 5;
P_abe.boxSeparation = 5;
P_abe.modelImageSampleFormat = AutomaticBackgroundExtractor.prototype.f32;
P_abe.abeDownsample = 2.00;
P_abe.writeSampleBoxes = false;
P_abe.justTrySamples = false;
P_abe.targetCorrection = AutomaticBackgroundExtractor.prototype.Subtract;
P_abe.normalize = false;
P_abe.discardModel = true;
P_abe.replaceTarget = true;
P_abe.correctedImageId = "";
P_abe.correctedImageSampleFormat = AutomaticBackgroundExtractor.prototype.SameAsTarget;
P_abe.verboseCoefficients = false;
P_abe.compareModel = false;
P_abe.compareFactor = 10.00;

var P_fr = new FastRotation;
P_fr.mode = FastRotation.prototype.HorizontalMirror;
P_fr.noGUIMessages = true;

var P_ir = new IntegerResample;
P_ir.zoomFactor = -2;
P_ir.downsamplingMode = IntegerResample.prototype.Average;
P_ir.xResolution = 72.000;
P_ir.yResolution = 72.000;
P_ir.metric = false;
P_ir.forceResolution = false;
P_ir.gammaCorrection = false;
P_ir.noGUIMessages = true;

let window = ImageWindow.open("{input_filename}");
window[0].show();

P_abe.executeOn(window[0].currentView);

P_bxt.executeOn(window[0].currentView);

if ("{rig}" == "C11") {{
  P_ir.executeOn(window[0].currentView);
}}

if ("{rig}" == "RC71") {{
  P_fr.executeOn(window[0].currentView);
}}

window[0].saveAs("{xisf_filename}", false, false, true, false);

for (let i = 0; i < window.length; ++i) {{
  window[i].forceClose();
}}
"""

# def add_xisf_fits_header(filename, data):
#   xisf = XISF(filename)
#   file_meta = xisf.get_file_metadata()
#   ims_meta = xisf.get_images_metadata()

#   fits_keywords = ims_meta[0]['FITSKeywords']
#   for k, v in data.items():
#     fits_keywords[k] = [{'value': str(v), 'comment': ''}]
#     print(f"Added FITS keyword {k} = {fits_keywords[k]}")

#   im_data = xisf.read_image(0)
#   XISF.write(filename,
#              im_data,
#              creator_app="PixInsight",
#              image_metadata=ims_meta[0],
#              xisf_metadata=file_meta,
#              codec='lz4hc',
#              shuffle=True)


@cache.memoize(tag='post_processing')
def run(rig, master):
  filename, ext = os.path.splitext(master)
  # fit_filename = filename + '_post' + '.fit'
  xisf_filename = filename + '_post' + '.xisf'

  if config.STACKING_TEST_MODE:
    print(f'In test mode; immediately returning {xisf_filename}')
    Path(xisf_filename).touch()
    return xisf_filename

  if os.path.exists(xisf_filename):
    os.remove(xisf_filename)

  with tempfile.TemporaryDirectory() as tmp_dir:
    script_filename = os.path.join(tmp_dir, "run_post_processing.js")
    with open(script_filename, "w") as f:
      script = post_process_script.format(input_filename=master,
                                          rig=rig,
                                          xisf_filename=xisf_filename)
      f.write(script)

    print(f'Running PixInsight post-processing on master {master}')
    ret = subprocess.run([
        '/Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight',
        f'--execute={script_filename}'
    ]).returncode

    while True:
      if os.path.exists(xisf_filename):
        print(f'PixInsight completed, result {xisf_filename}')
        break
      else:
        time.sleep(5)

    # center, footprint, wcs = plate_solving.solve_field(fit_filename)
    # if not wcs:
    #   print(f'Failed to plate-solve post-processed result')
    #   return None

    # with pyfits.open(io.BytesIO(wcs)) as fits:
    #   idl_keys = [
    #       "CD1_1", "CD1_2", "CD2_1", "CD2_2", "CRPIX1", "CRPIX2", "CRVAL1",
    #       "CRVAL2", "LONPOLE", "LATPOLE", "CUNIT1", "CUNIT2", "CTYPE1",
    #       "CTYPE2"
    #   ]
    #   idl_solution = {k: fits[0].header[k] for k in idl_keys}

    #   add_xisf_fits_header(xisf_filename, idl_solution)
    #   print(f'Wrote {len(idl_solution)} WCS FITS values into XISF '
    #         f'{xisf_filename}')

    # os.remove(fit_filename)

    return xisf_filename
