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


_REGISTRY: dict[tuple[str, type], Callable] = {}
_DISCOVERED: bool = False


def _discover_builtin_pipelines() -> None:
    """Import every user-pipeline `.py` file in `core/pipelines/`.

    Skips the registry/decorators support modules and any `test_*.py`.
    Failures are logged but don't abort discovery — one broken file
    must not knock out the others.

    Uses `importlib.reload` when the module is already in
    `sys.modules` so the `@pipeline` decoration re-runs and the
    registry is refreshed. Without this, a previous discovery (or any
    other import) would cache the module, and a subsequent call after
    the registry was cleared (e.g. by a test fixture) would silently
    skip re-registration.
    """
    import sys
    pipelines_dir = Path(__file__).parent
    skip = {"__init__.py", "registry.py", "decorators.py"}
    for path in sorted(pipelines_dir.glob("*.py")):
        if path.name in skip or path.name.startswith("test_"):
            continue
        module_name = f"pipelines.{path.stem}"
        try:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
            else:
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


def discover_pipelines() -> dict[tuple[str, type], Callable]:
    """Import every built-in and user pipeline file once. Subsequent
    calls are no-ops (use `refresh_pipelines()` to force a re-scan).
    Returns a snapshot of the registry, keyed by `(display_name, processes)`."""
    global _DISCOVERED
    if not _DISCOVERED:
        _discover_builtin_pipelines()
        _discover_user_pipelines()
        _DISCOVERED = True
    return dict(_REGISTRY)


def refresh_pipelines() -> dict[tuple[str, type], Callable]:
    """Force a re-scan of the pipeline directories. Useful when a user
    has just added or edited a pipeline file; cheaper than restarting
    the process. Note: Python's import cache means edits to a
    previously-loaded built-in won't be re-read; user files always
    re-exec, since they load by file location each time."""
    global _DISCOVERED
    _DISCOVERED = False
    return discover_pipelines()


def get_pipeline(display_name: str, processes: type) -> Callable:
    """Look up a pipeline entry point by `(display_name, processes)`.

    `processes` is the entity class the user is queuing — the
    `Stack` / `Target` / `Mosaic` class. Triggers
    `discover_pipelines()` on first call. Raises `KeyError` with a
    list of available `(name, class.__name__)` pairs when unknown —
    so a typo or a pipeline that doesn't support this entity type
    fails loudly rather than silently falling through to a default
    builder.
    """
    if not _DISCOVERED:
        discover_pipelines()
    key = (display_name, processes)
    if key not in _REGISTRY:
        available = sorted(
            (n, cls.__name__) for (n, cls) in _REGISTRY.keys()
        )
        raise KeyError(
            f"No pipeline registered with display name {display_name!r} "
            f"for class {processes.__name__!r}. Available: {available}"
        )
    return _REGISTRY[key]


def registered_pipeline_names() -> set[str]:
    """Set of distinct pipeline display names — the values the cart's
    pipeline dropdown should offer. A name may have multiple entry
    points (one per class it processes); this collapses them into
    the user-facing name list."""
    if not _DISCOVERED:
        discover_pipelines()
    return {name for name, _ in _REGISTRY.keys()}


def registered_pipelines() -> dict[tuple[str, type], Callable]:
    """Snapshot of the full registry, keyed by `(name, processes)`.
    Triggers discovery on first call. Most callers want
    `registered_pipeline_names()` (just the names); this returns the
    full per-class mapping for the cart dispatcher."""
    if not _DISCOVERED:
        discover_pipelines()
    return dict(_REGISTRY)
