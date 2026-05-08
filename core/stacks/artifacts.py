import math
import os
from pathlib import Path

from stacks.composite_stack import BaseStack, Stack
from catalogs import position_string
from file_io.types import XisfFilePath, FitsFilePath


def stack_filename(stack: BaseStack) -> Path:
    if "bias" in stack.image_type:
        return XisfFilePath(
            f"bias_{stack.stable_hash}_{stack.camera}_"
            f"{stack.mean_temperature:.0f}C_{stack.stable_hash}.xisf"
        )
    elif "dark" in stack.image_type:
        return XisfFilePath(
            f"dark_{stack.stable_hash}_{stack.camera}_"
            f"{stack.mean_temperature:.0f}C_{stack.exposure:.0f}s_"
            f"{stack.stable_hash}.xisf"
        )
    elif "flat" in stack.image_type:
        return XisfFilePath(
            f"flat_{stack.stable_hash}_{stack.camera}_{stack.filter}_"
            f"{stack.stable_hash}.xisf"
        )
    elif isinstance(stack, Stack) and "light" in stack.image_type:
        ra_dec_string = position_string.ra_dec_string(*stack.center)
        return XisfFilePath(
            f"light_{stack.stable_hash}_{stack.name}_{ra_dec_string}_"
            f"{stack.camera}_{stack.image_scale:.02f}aspp_"
            f"{math.degrees(stack.rotation_angle):.2f}deg_{stack.filter}.xisf"
        )

    assert False, f'unknown image_type "{stack.image_type}"'


def stack_filename_fits(stack: BaseStack) -> Path:
    """FITS variant of stack_filename for the OSS pipeline."""
    xisf_path = stack_filename(stack)
    return FitsFilePath(Path(str(xisf_path)).with_suffix(".fits"))
