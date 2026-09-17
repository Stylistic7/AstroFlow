"""The downscaled pipelines' factor. The whole-pipeline speedup and the
docs that quote it depend on this constant."""
from previews.types import DOWNSCALED_MASTER, DOWNSCALE_FACTORS


def test_downscaled_factor_is_8():
    """8x downscale = 1/64 the pixel count. Tuning it means updating both
    this test and the speedup ratio the pipeline docstrings quote."""
    assert DOWNSCALE_FACTORS[DOWNSCALED_MASTER] == 8
