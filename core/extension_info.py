"""Who wrote a pipeline or finish, which version it is, what it does, and
where it lives — the facts the Pipelines and Finishes pages list
(Stylistic7, 2026-10-03).

Built-ins ship inside the AstroFlow tree: their author is AstroFlow and
their version is AstroFlow's (pyproject version + commit). Installed ones
load from `~/AstroFlow/pipelines/` or `~/AstroFlow/finishes/` and carry
whatever their author declared (`author=`, `version=`, `description=` on
`@pipeline` / `@finish`); undeclared facts stay None rather than guessed.
"""
from __future__ import annotations

import inspect
import subprocess
import tomllib
from functools import cache
from pathlib import Path

_CORE = Path(__file__).resolve().parent


@cache
def astroflow_version() -> str:
    """`0.1 (3f4ec514)`: the project version plus the commit it runs."""
    repo = _CORE.parent
    version = tomllib.loads((repo / "pyproject.toml").read_text())["project"]["version"]
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return version
    return f"{version} ({sha})"


def first_paragraph(doc: "str | None") -> "str | None":
    """A docstring's opening paragraph on one line — its summary."""
    if not doc or not doc.strip():
        return None
    return " ".join(inspect.cleandoc(doc).split("\n\n")[0].split())


def describe(fn, *, doc: "str | None", author: "str | None" = None,
             version: "str | None" = None,
             description: "str | None" = None) -> dict:
    """The listing facts for the code that defines `fn`. `doc` is the
    docstring to summarise when no `description` was declared."""
    source = Path(inspect.getsourcefile(fn)).resolve()
    line = inspect.getsourcelines(fn)[1]
    builtin = source.is_relative_to(_CORE)
    return {
        "author": author or ("AstroFlow" if builtin else None),
        "version": version or (astroflow_version() if builtin else None),
        "description": description or first_paragraph(doc),
        "origin": "built-in" if builtin else "installed",
        "location": f"{source}:{line}",
    }
