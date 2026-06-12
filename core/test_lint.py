"""Run ruff over `core/` as part of the test suite.

Catches the class of bug that bit us in `stacks/post_processing/pixinsight.py`
— `Path` referenced inside `if not mock_pixinsight:` was undefined, but
every test runs with `mock_pixinsight=True` so the branch was never
executed and the NameError shipped to production.

Rule set is pinned in `pyproject.toml` (`[tool.ruff.lint]`). Add high-
signal rules there as the suite tolerates them; this test imposes no
policy of its own — it just enforces what ruff is already configured to
check.
"""
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = REPO_ROOT / "core"


def test_ruff_lint_passes():
    """Fails on any rule selected in `[tool.ruff.lint]`. Run locally:
        ruff check core/
    """
    ruff = shutil.which("ruff")
    if ruff is None:
        pytest.skip("ruff not installed (uv pip install ruff)")

    proc = subprocess.run(
        [ruff, "check", "--no-cache", str(CORE_DIR)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"ruff check failed (exit {proc.returncode}):\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
