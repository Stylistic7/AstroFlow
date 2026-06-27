"""The XISF reader runs `ast.literal_eval` on every FITS keyword value, so an
all-digit STBLHASH round-trips as an int. A cal stack whose stable_hash is all
digits then broke the str-keyed `cal_master_by_id` lookup in `stack_files`
(`KeyError: '4900669000273492'`). Identifiers must be keyed/compared as strings;
the fix is `str()` at the read/lookup sites. Ray-free — uses real XISF I/O.
"""
import tempfile
from pathlib import Path

import numpy as np

from file_io.xisf_file_io import write_xisf_image
from file_io.generic import read_header


def test_all_digit_stblhash_coerces_to_int_and_str_recovers():
    path = Path(tempfile.mkdtemp()) / "m.xisf"
    write_xisf_image(path, np.zeros((4, 4, 1), np.float32), {"STBLHASH": "4900669000273492"})
    raw = read_header(path)["STBLHASH"]
    # The hazard: an all-digit identifier comes back as an int, not the string.
    assert isinstance(raw, int) and raw == 4900669000273492
    # The fix applied at the use sites: str() recovers a stable string key.
    assert str(raw) == "4900669000273492"


def test_hex_stblhash_stays_str():
    path = Path(tempfile.mkdtemp()) / "m.xisf"
    write_xisf_image(path, np.zeros((4, 4, 1), np.float32), {"STBLHASH": "5eedb9c618f32f58"})
    # Hex hashes contain letters, so literal_eval fails → the string survives.
    assert read_header(path)["STBLHASH"] == "5eedb9c618f32f58"
