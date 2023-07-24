import subprocess
import sys
import tempfile
import os
from pathlib import Path
from multiprocessing import Pool
import astropy.io.fits as pyfits

save_as_fits_script = """
let window = ImageWindow.open("{input_filename}");
window[0].show();

window[0].saveAs("{fit_filename}", false, false, true, false);

for (let i = 0; i < window.length; ++i) {{
  window[i].forceClose();
}}
"""


def solve_field(filename):
  with tempfile.TemporaryDirectory() as temp_dir:
    dest_filename = os.path.join(temp_dir, os.path.basename(filename))
    os.symlink(filename, dest_filename)

    try:
      command = [
          "solve-field", "--overwrite", "--no-plots", "--new-fits", "none",
          "--no-background-subtraction", "--guess-scale", "--objs", "1000",
          dest_filename
      ]
      ret = subprocess.run(command, capture_output=True, timeout=120)
    except:
      return None, None

    stdout = ret.stdout.decode("utf-8")

    wcs_file = Path(dest_filename).with_suffix('.wcs')
    if os.path.exists(wcs_file):
      wcs = open(wcs_file, 'rb').read()
    else:
      # TODO: Emit warnings through Metaflow's reporting system.
      print(f'Could not plate-solve file {filename}')
      wcs = None

    return wcs, stdout


def solve_fields(filenames):
  with Pool() as p:
    solutions = p.map(solve_field, filenames)
    return {f: s for f, s in zip(filenames, solutions) if s[0] != None}


if __name__ == '__main__':
  wcs, stdout = solve_field(sys.argv[1])
