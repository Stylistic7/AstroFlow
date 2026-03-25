import math
import os
from pathlib import Path

from stacks.simple_stack import Run
from stacks.composite_stack import CompositeStack
from catalogs import position_string
from file_io.types import XisfFilePath


def stack_filename(stack: Run | CompositeStack) -> Path:
    def exposure_string(exposures):
        return "_".join([f"{x}s" for x in exposures])

    if isinstance(stack, Run) and "bias" in stack.image_type:
        return XisfFilePath(
            f"bias_{stack.stable_hash}_{stack.camera}_"
            f"{stack.mean_temperature:.0f}C_{stack.stable_hash}.xisf"
        )
    elif isinstance(stack, Run) and "dark" in stack.image_type:
        return XisfFilePath(
            f"dark_{stack.stable_hash}_{stack.camera}_"
            f"{stack.mean_temperature:.0f}C_{exposure_string(stack.exposure)}_"
            f"{stack.stable_hash}.xisf"
        )
    elif isinstance(stack, Run) and "flat" in stack.image_type:
        return XisfFilePath(
            f"flat_{stack.stable_hash}_{stack.camera}_{stack.filter}_"
            f"{stack.stable_hash}.xisf"
        )
    elif isinstance(stack, CompositeStack) and "light" in stack.image_type:
        ra_dec_string = position_string.ra_dec_string(*stack.center)
        return XisfFilePath(
            f"light_{stack.stable_hash}_{stack.name}_{ra_dec_string}_"
            f"{stack.camera}_{stack.image_scale:.02f}aspp_"
            f"{math.degrees(stack.rotation_angle):.2f}deg_{stack.filter}.xisf"
        )

    assert False, f'unknown image_type "{stack.image_type}"'
