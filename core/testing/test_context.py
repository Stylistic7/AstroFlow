from pathlib import Path
import tempfile
import datetime
import numpy as np
from dataclasses import dataclass
from typing import Any

from context import context
from context import get_cache
from customization.settings import Settings


@dataclass(frozen=True)
class TestContext:
    input_dir_obj: Any
    output_dir_obj: Any


def make_test_context() -> tuple[
    tempfile.TemporaryDirectory, tempfile.TemporaryDirectory
]:
    context.clear()

    context.stacks = {}
    context.targets = {}
    context.mosaics = {}

    context.mock_pixinsight = True
    context.mock_plate_solving = True

    input_dir_obj = tempfile.TemporaryDirectory()
    context.input_dirs = [Path(input_dir_obj.name)]

    output_dir_obj = tempfile.TemporaryDirectory()
    context.output_dir = Path(output_dir_obj.name)

    context.cosine_similarity_threshold = 0.99999
    context.angular_tolerance = np.radians(10.0)
    context.min_component_members = 3
    context.max_run_gap = datetime.timedelta(hours=8)
    context.min_frames_in_stack = 3
    context.subframe_thumbnail_width = 1200
    context.master_thumbnail_width = 2160
    context.subframe_mosaic_samples = 500

    context.settings = Settings()

    return TestContext(input_dir_obj, output_dir_obj)
