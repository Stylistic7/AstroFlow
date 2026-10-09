"""post_stack_calibrate writes its result as XISF from a FITS master (Siril's)
too: pixels swapped in, descriptive keywords kept, the FITS WCS dropped."""
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

from file_io.generic import read_header, read_image
from stacks.post_processing.graxpert_stack import _write_pixels_preserving_metadata


def test_a_fits_master_becomes_an_xisf_with_its_keywords():
    d = Path(tempfile.mkdtemp())
    src = d / "master.fit"
    hdr = fits.Header({"FILTER": "OIII", "STBLHASH": "ab12cd34ef56ab78", "STACKCNT": 3,
                       "CTYPE1": "RA---TAN-SIP", "CRPIX1": 2.0, "A_ORDER": 2})
    pixels = np.arange(12, dtype=np.float32).reshape(3, 4)
    fits.writeto(src, pixels, hdr)
    _write_pixels_preserving_metadata(src, d / "out.xisf", pixels * 2)
    out = np.asarray(read_image(d / "out.xisf"), np.float32).squeeze()
    h = read_header(d / "out.xisf")
    assert np.array_equal(out, pixels * 2)
    assert h["FILTER"] == "OIII" and str(h["STBLHASH"]) == "ab12cd34ef56ab78"
    assert "CTYPE1" not in h and "CRPIX1" not in h and "A_ORDER" not in h
