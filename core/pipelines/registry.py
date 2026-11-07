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

    Never reloads a module that is already imported: a reload rebinds its
    `@node`s, and a graph already holding the old ones can then only be
    pickled by value, which `dependency_graph.serialization` cannot read
    back. `run_pipeline` got exactly that: unpickling its traced graph
    imported `pipelines.time_lapse`, and its first `get_pipeline` reloaded
    it, so returning the graph ended the pass (2026-11-07).
    """
    pipelines_dir = Path(__file__).parent
    skip = {"__init__.py", "registry.py", "decorators.py"}
    for path in sorted(pipelines_dir.glob("*.py")):
        if path.name in skip or path.name.startswith("test_"):
            continue
        try:
            importlib.import_module(f"pipelines.{path.stem}")
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
    """Set of distinct USER-FACING pipeline display names — the values the
    cart's pipeline dropdown should offer. A name may have multiple entry
    points (one per class it processes); this collapses them into the
    user-facing name list. Pipelines registered `listed=False` (Subframe,
    the implicit always-run preview) are excluded here — and ONLY here:
    every other consumer (catalog, sweep, trace) sees the full registry."""
    if not _DISCOVERED:
        discover_pipelines()
    return {
        name for (name, _), runner in _REGISTRY.items()
        if getattr(runner, "_pipeline_listed", True)
    }


def registered_pipelines() -> dict[tuple[str, type], Callable]:
    """Snapshot of the full registry, keyed by `(name, processes)`.
    Triggers discovery on first call. Most callers want
    `registered_pipeline_names()` (just the names); this returns the
    full per-class mapping for the cart dispatcher."""
    if not _DISCOVERED:
        discover_pipelines()
    return dict(_REGISTRY)


_PROCESS_ORDER = ("Stack", "Target", "Mosaic")


def pipeline_listing() -> list[dict]:
    """One row per pipeline name for the Pipelines page: author, version,
    description, built-in or installed, and the file (and line) defining
    it, plus the entity classes it processes. Unlisted (app-internal)
    pipelines are included and marked, since they are present."""
    from extension_info import describe

    if not _DISCOVERED:
        discover_pipelines()
    by_name: dict[str, list] = {}
    for (name, cls), runner in _REGISTRY.items():
        by_name.setdefault(name, []).append((cls.__name__, runner))
    rows = []
    for name, entries in sorted(by_name.items()):
        entries.sort(key=lambda e: _PROCESS_ORDER.index(e[0])
                     if e[0] in _PROCESS_ORDER else len(_PROCESS_ORDER))
        fn = entries[0][1]._function
        declared = {k: next((r._pipeline_about[k] for _, r in entries
                             if getattr(r, "_pipeline_about", {}).get(k)), None)
                    for k in ("author", "version", "description")}
        rows.append({
            "name": name,
            # The pipeline FILE's docstring describes the pipeline; each
            # entry point's own docstring describes only one level of it.
            **describe(fn, doc=fn.__globals__.get("__doc__"), **declared),
            "processes": [c for c, _ in entries],
            "listed": all(getattr(r, "_pipeline_listed", True) for _, r in entries),
        })
    return rows
