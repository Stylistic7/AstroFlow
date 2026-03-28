"""Lazy execution recipe for PixInsight operations.

A Recipe collects PixInsight operations (steps) without executing them.
Execution is deferred until an Image.save_to() call forces materialization.
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pix.image import Image


class PixInsightError(Exception):
    """A PixInsight task reported an error in its console output."""
    pass


@dataclass
class Step:
    """A single PixInsight operation in a recipe.

    Steps come in two flavors:
    - Inline JavaScript: set `javascript` to the JS code.
    - External script: set `script_filename` to the .js path, with optional
      `parameters` (passed via -p flags) and `arguments` (via -a flags).

    error_patterns: compiled regexes matched against the PixInsight console log
    after execution. If any pattern matches, a PixInsightError is raised.

    success_pattern: if set, this regex MUST match the log or a PixInsightError
    is raised. Use this for tasks where intermediate errors are acceptable as
    long as the final output is produced (e.g. ImageSolver).
    """
    name: str = ""
    javascript: str | None = None
    script_filename: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    arguments: dict[str, Any] = field(default_factory=dict)
    error_patterns: list[re.Pattern] = field(default_factory=list)
    success_pattern: re.Pattern | None = None


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
            prefix=f"pix-{self._recipe_id}-"
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

    def _write_step_js(self, step: Step) -> Path | None:
        """Write inline JavaScript to a temp file. Returns the path."""
        if step.javascript is None:
            return None
        js_path = self.tmp_dir / f"{uuid.uuid4().hex[:12]}.js"
        js_path.write_text(step.javascript)
        return js_path

    def _scp_command_for_step(self, step: Step) -> str:
        """Generate the .scp run command for a step."""
        if step.script_filename:
            filename = step.script_filename
        else:
            js_path = self._write_step_js(step)
            if js_path is None:
                return ""
            filename = str(js_path)

        args = " ".join(
            f'-a="{k}={v}"' for k, v in step.arguments.items()
        )
        params = " ".join(
            f'-p="{k},{v}"' for k, v in step.parameters.items()
        )
        parts = ["run", "-x"]
        if args:
            parts.append(args)
        if params:
            parts.append(params)
        parts.append(f'"{filename}"')
        return " ".join(parts)

    def to_javascript(self, save: dict[Image, Path] | None = None) -> str:
        """Generate the inline JavaScript for steps that use it.

        For recipes with only inline JS steps, this returns the complete
        script. For mixed recipes (inline + external scripts), use
        to_scp() instead which handles both.
        """
        from pix.tasks.open_images import generate_open_images_js
        from pix.tasks.save_images import generate_save_images_js

        log_file = self.tmp_dir / f"{self._recipe_id}.log"

        lines = []
        lines.append('console.beginLog();')
        lines.append('')

        # Open input images
        lines.append(generate_open_images_js(self.input_filenames))
        lines.append('')

        # All processing steps (inline JS only)
        for step in self.steps:
            if step.javascript:
                if step.name:
                    lines.append(f'// {step.name}')
                lines.append(step.javascript)
                lines.append('')

        # Save outputs
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

    def _has_script_steps(self) -> bool:
        """Does this recipe contain any external script steps?"""
        return any(s.script_filename for s in self.steps)

    def to_scp(self, save: dict[Image, Path] | None = None) -> str:
        """Generate the complete .scp script for this recipe.

        Handles both inline JavaScript and external script steps.
        """
        from pix.tasks.open_images import generate_open_images_js
        from pix.tasks.save_images import generate_save_images_js

        log_file = self.tmp_dir / f"{self._recipe_id}.log"

        if not self._has_script_steps():
            # All inline JS — generate a single .js file and run it
            js = self.to_javascript(save=save)
            js_path = self.tmp_dir / f"{self._recipe_id}_main.js"
            js_path.write_text(js)
            return f'run -x "{js_path}"'

        # Mixed recipe: generate separate commands for each step
        scp_lines = []

        # Open images as first JS step
        open_js = (
            'console.beginLog();\n\n'
            + generate_open_images_js(self.input_filenames)
        )
        open_path = self.tmp_dir / f"{self._recipe_id}_open.js"
        open_path.write_text(open_js)
        scp_lines.append(f'run -x "{open_path}"')

        # Each processing step
        for step in self.steps:
            cmd = self._scp_command_for_step(step)
            if cmd:
                scp_lines.append(cmd)

        # Save outputs + capture log
        save_targets: dict[str, Path] = {}
        if save:
            for image, path in save.items():
                save_targets[image.view_id] = path
        else:
            for image in self.outputs:
                save_targets[image.view_id] = self._output_path_in_tmp(image)

        save_js = generate_save_images_js(save_targets)
        save_js += f'\nFile.writeTextFile("{log_file}", console.endLog().toString());'
        save_path = self.tmp_dir / f"{self._recipe_id}_save.js"
        save_path.write_text(save_js)
        scp_lines.append(f'run -x "{save_path}"')

        return '\n'.join(scp_lines)

    def materialize(self, mock: bool = False) -> None:
        """Execute the recipe, saving all outputs to the temp directory.

        After materialization, individual Image.save_to() calls just
        copy from the temp dir — no re-execution needed.
        """
        if self.is_materialized:
            return

        if mock:
            src = self.input_filenames[0]
            for image in self.outputs:
                dest = self._output_path_in_tmp(image)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
        else:
            save_all = {
                image: self._output_path_in_tmp(image)
                for image in self.outputs
            }
            scp = self.to_scp(save=save_all)
            self._run_pixinsight(scp)

            log_file = self.tmp_dir / f"{self._recipe_id}.log"
            if log_file.exists():
                self.log = log_file.read_text()

            if self.log:
                self._emit_log()
                self._check_for_errors()

        self.is_materialized = True

    def _emit_log(self) -> None:
        """Emit PixInsight console output through structlog for capture."""
        import structlog
        logger = structlog.get_logger()
        for line in self.log.splitlines():
            if line.strip():
                logger.info(line)

    def _check_for_errors(self) -> None:
        """Check PixInsight console log against each step's error/success patterns."""
        for step in self.steps:
            for pattern in step.error_patterns:
                match = pattern.search(self.log)
                if match:
                    raise PixInsightError(
                        f"{step.name}: {match.group(0)}"
                    )
            if step.success_pattern and not step.success_pattern.search(self.log):
                raise PixInsightError(
                    f"{step.name} did not produce expected output"
                )

    def _run_pixinsight(self, scp_content: str) -> None:
        """Execute a .scp script in a real PixInsight instance."""
        from pix.runner import run_scp
        run_scp(scp_content, self.tmp_dir)

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
