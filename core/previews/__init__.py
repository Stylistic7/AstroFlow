from typing import NewType

from .operations import array_to_png, autostretch, resize_array
from .stack_thumbnails import scatter_stack_thumbnails, create_stack_subframe_thumbnails
from .flat_correction import create_flat_correction_thumbnails
from .calibration import calibrate_image

PreviewHash = NewType('PreviewHash', str)
