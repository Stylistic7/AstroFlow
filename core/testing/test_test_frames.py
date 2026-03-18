import datetime

from plate_solving import PlateSolution
from testing import make_synthetic_frames


def test_one_perfectly_centered_image(test_context):
    make_synthetic_frames(number=1, prefix="foo", dither=0)

    expected_fits_headers = {
        "foo_00000.fit": {
            "DATE-OBS": datetime.datetime(2023, 12, 6, 16, 29, 43),
            "IMAGETYP": "Light",
            "EXPOSURE": 300,
            "CRVAL1": 100.0,
            "CRVAL2": 45.0,
        }
    }

    # TODO: Actually compare these...
