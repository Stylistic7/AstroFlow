import subprocess
import sys
import tempfile
import os
from pathlib import Path


def solve_field(filename):
  with tempfile.TemporaryDirectory() as temp_dir:
    dest_filename = os.path.join(temp_dir, os.path.basename(filename))
    os.symlink(filename, dest_filename)

    command = [
        "solve-field", "--overwrite", "--no-plots", "--new-fits", "none",
        "--no-background-subtraction", "--guess-scale", dest_filename
    ]
    ret = subprocess.run(command, capture_output=True)

    stdout = ret.stdout.decode("utf-8")

    wcs_file = Path(dest_filename).with_suffix('.wcs')
    if os.path.exists(wcs_file):
      wcs = open(wcs_file).read()
    else:
      wcs = None

    return wcs, stdout


if __name__ == '__main__':
  wcs, stdout = solve_field(sys.argv[1])
