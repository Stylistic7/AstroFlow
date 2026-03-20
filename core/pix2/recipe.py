"""Lazy execution recipe for PixInsight operations.

A Recipe collects PixInsight operations (steps) without executing them.
Execution is deferred until an Image.save_to() call forces materialization.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pix2.image import Image


@dataclass
class Step:
    """A single PixInsight operation in a recipe."""
    javascript: str
    name: str = ""


class Recipe:
    """A lazy collection of PixInsight operations.

    The recipe accumulates steps as tasks are applied to images.
    Nothing is executed until save_to() is called on an output image.
    """

    def __init__(self, input_filenames: list[Path]):
        self.input_filenames = input_filenames
        self.steps: list[Step] = []
        self.outputs: list[Image] = []
        self.is_materialized = False
        self.log: str | None = None
        self._recipe_id = uuid.uuid4().hex[:12]
        self._tmp_dir_obj = tempfile.TemporaryDirectory(
            prefix=f"pix2-{self._recipe_id}-"
        )

    @property
    def tmp_dir(self) -> Path:
        return Path(self._tmp_dir_obj.name)

    def add_step(self, step: Step) -> None:
        self.steps.append(step)

    def register_output(self, image: Image) -> None:
        self.outputs.append(image)

    def _output_path_in_tmp(self, image: Image) -> Path:
        """The temp file path where this image's view will be saved."""
        return self.tmp_dir / f"{image.view_id}.xisf"

    def to_javascript(self, save: dict[Image, Path] | None = None) -> str:
        """Generate the complete PixInsight JavaScript for this recipe.

        Args:
            save: Mapping of Image → output path. If None, saves all
                  registered outputs to the temp directory.
        """
        from pix2.tasks.open_images import generate_open_images_js
        from pix2.tasks.save_images import generate_save_images_js

        log_file = self.tmp_dir / f"{self._recipe_id}.log"

        lines = []
        lines.append('console.beginLog();')
        lines.append('')

        # Open input images
        lines.append(generate_open_images_js(self.input_filenames))
        lines.append('')

        # All processing steps
        for step in self.steps:
            if step.name:
                lines.append(f'// {step.name}')
            lines.append(step.javascript)
            lines.append('')

        # Save outputs — either to requested paths or to temp dir
        save_targets: dict[str, Path] = {}
        if save:
            for image, path in save.items():
                save_targets[image.view_id] = path
        else:
            for image in self.outputs:
                save_targets[image.view_id] = self._output_path_in_tmp(image)

        lines.append(generate_save_images_js(save_targets))
        lines.append('')

        # Capture console log
        lines.append(
            f'File.writeTextFile("{log_file}", console.endLog().toString());'
        )

        return '\n'.join(lines)

    def materialize(self, mock: bool = False) -> None:
        """Execute the recipe, saving all outputs to the temp directory.

        After materialization, individual Image.save_to() calls just
        copy from the temp dir — no re-execution needed.
        """
        if self.is_materialized:
            return

        if mock:
            # Mock: copy the first input file to each output location
            src = self.input_filenames[0]
            for image in self.outputs:
                dest = self._output_path_in_tmp(image)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
        else:
            # Real execution: generate JS, run PixInsight, capture log
            save_all = {
                image: self._output_path_in_tmp(image)
                for image in self.outputs
            }
            js = self.to_javascript(save=save_all)
            self._run_pixinsight(js)

            log_file = self.tmp_dir / f"{self._recipe_id}.log"
            if log_file.exists():
                self.log = log_file.read_text()

        self.is_materialized = True

    def _run_pixinsight(self, javascript: str) -> None:
        """Execute JavaScript in a real PixInsight instance."""
        from pix2.runner import run_javascript
        run_javascript(javascript, self.tmp_dir)

    def copy_output(self, image: Image, dest: Path) -> None:
        """Copy a materialized output to the requested path."""
        src = self._output_path_in_tmp(image)
        if not src.exists():
            raise FileNotFoundError(
                f"Output for view '{image.view_id}' not found in recipe "
                f"temp dir. Was the recipe materialized?"
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
