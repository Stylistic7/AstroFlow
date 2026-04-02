from pathlib import Path
import tempfile
from dataclasses import dataclass
from typing import Any

from context import context
from context import get_cache
from decisions.settings import Settings, ClusteringSettings, DevelopmentSettings, PreviewSettings


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

    input_dir_obj = tempfile.TemporaryDirectory()
    context.input_dirs = [Path(input_dir_obj.name)]

    output_dir_obj = tempfile.TemporaryDirectory()
    context.output_dir = Path(output_dir_obj.name)

    context.settings = Settings(
        clustering_settings=ClusteringSettings(
            min_frames_in_stack=3,
        ),
        development_settings=DevelopmentSettings(
            mock_pixinsight=True,
            dangerously_allow_missing_files=True,
            test_mode=True,
        ),
        preview_settings=PreviewSettings(
            subframe_thumbnail_width=1200,
            master_thumbnail_width=2160,
            subframe_mosaic_samples=500,
        ),
    )

    return TestContext(input_dir_obj, output_dir_obj)
