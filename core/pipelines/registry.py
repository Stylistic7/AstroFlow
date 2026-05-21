"""Pipeline registry: a tiny dict + lookup, populated by `@pipeline(...)`
decorations at module-import time, plus a lazy directory scan that
triggers those imports on first lookup.
"""
import importlib
import importlib.util
from pathlib import Path
from typing import Callable

import structlog

log = structlog.get_logger()


_REGISTRY: dict[str, Callable] = {}
_DISCOVERED: bool = False


def _discover_builtin_pipelines() -> None:
    """Import every user-pipeline `.py` file in `core/pipelines/`.

    Skips the registry/decorators support modules and any `test_*.py`.
    Failures are logged but don't abort discovery — one broken file
    must not knock out the others.
    """
    pipelines_dir = Path(__file__).parent
    skip = {"__init__.py", "registry.py", "decorators.py"}
    for path in sorted(pipelines_dir.glob("*.py")):
        if path.name in skip or path.name.startswith("test_"):
            continue
        module_name = f"pipelines.{path.stem}"
        try:
            importlib.import_module(module_name)
            log.debug("Loaded built-in pipeline file", path=str(path))
        except Exception as e:
            log.warning(
                "Failed to import built-in pipeline file",
                path=str(path), error=str(e),
            )


def _discover_user_pipelines() -> None:
    """Import every `.py` file in `~/AstroFlow/pipelines/`.

    Same failure semantics as the built-in scan. User files load via
    a custom module name (`_user_pipelines.<stem>`) so they don't
    collide with the `pipelines.*` package; their `@pipeline(...)`
    side effects still hit the same global registry.
    """
    user_dir = Path.home() / "AstroFlow" / "pipelines"
    if not user_dir.is_dir():
        return
    for path in sorted(user_dir.glob("*.py")):
        if path.name.startswith("_") or path.name.startswith("test_"):
            continue
        spec = importlib.util.spec_from_file_location(
            f"_user_pipelines.{path.stem}", path,
        )
        if spec is None or spec.loader is None:
            log.warning("Cannot load user pipeline file", path=str(path))
            continue
        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            log.info("Loaded user pipeline file", path=str(path))
        except Exception as e:
            log.warning(
                "Failed to import user pipeline file",
                path=str(path), error=str(e),
            )


def discover_pipelines() -> dict[str, Callable]:
    """Import every built-in and user pipeline file once. Subsequent
    calls are no-ops (use `refresh_pipelines()` to force a re-scan).
    Returns a snapshot of the registry."""
    global _DISCOVERED
    if not _DISCOVERED:
        _discover_builtin_pipelines()
        _discover_user_pipelines()
        _DISCOVERED = True
    return dict(_REGISTRY)


def refresh_pipelines() -> dict[str, Callable]:
    """Force a re-scan of the pipeline directories. Useful when a user
    has just added or edited a pipeline file; cheaper than restarting
    the process. Note: Python's import cache means edits to a
    previously-loaded built-in won't be re-read; user files always
    re-exec, since they load by file location each time."""
    global _DISCOVERED
    _DISCOVERED = False
    return discover_pipelines()


def get_pipeline(display_name: str) -> Callable:
    """Look up a pipeline by its `@pipeline("…")` display name.

    Triggers `discover_pipelines()` on first call. Raises `KeyError`
    with a list of available names when unknown — so a typo in
    a shopping-cart-entry `pipeline_name` fails loudly rather than
    silently falling through to the default builder.
    """
    if not _DISCOVERED:
        discover_pipelines()
    if display_name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise KeyError(
            f"No pipeline registered with display name {display_name!r}. "
            f"Available: {available}"
        )
    return _REGISTRY[display_name]


def registered_pipelines() -> dict[str, Callable]:
    """Snapshot of the registry. Triggers discovery on first call."""
    if not _DISCOVERED:
        discover_pipelines()
    return dict(_REGISTRY)
