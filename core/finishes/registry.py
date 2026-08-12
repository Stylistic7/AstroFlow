"""Finish discovery — mirror of pipelines/registry.py.

Stock finishes come from `finishes.stock` (reloaded on refresh so the
decorator side-effects re-run after a registry clear); user finishes from
`~/AstroFlow/finishes/*.py`, loaded by file location under a synthetic
package so they can't collide with real modules. One broken user file logs
a warning and never aborts the scan. Re-registration silently overrides —
a user file replaces a stock finish by reusing its name + primary input.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import structlog

log = structlog.get_logger()

# (name, primary_input) → Finish
_REGISTRY: dict = {}
_DISCOVERED = False


def register_finish(f) -> None:
    _REGISTRY[(f.name, f.inputs[0])] = f


def _discover_builtin() -> None:
    mod = "finishes.stock"
    if mod in sys.modules:
        importlib.reload(sys.modules[mod])
    else:
        importlib.import_module(mod)


def _discover_user() -> None:
    user_dir = Path.home() / "AstroFlow" / "finishes"
    if not user_dir.is_dir():
        return
    for path in sorted(user_dir.glob("*.py")):
        if path.name.startswith("_") or path.name.startswith("test_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(
                f"_user_finishes.{path.stem}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            log.info("Loaded user finish file", path=str(path))
        except Exception as e:
            log.warning("Failed to import user finish file",
                        path=str(path), error=str(e))


def discover_finishes() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    _discover_builtin()
    _discover_user()
    _DISCOVERED = True


def refresh_finishes() -> None:
    """Re-scan everything — the author edit loop (`?refresh=1` on the
    endpoint). User files always re-exec (loaded by location); the stock
    module is explicitly reloaded."""
    global _DISCOVERED
    _REGISTRY.clear()
    _DISCOVERED = False
    discover_finishes()


def finish_specs() -> list:
    discover_finishes()
    return [f.spec() for f in _REGISTRY.values()]


def get_finish(name: str, primary: "str | None" = None):
    """Look up by name (+ primary input when disambiguating alt-primary
    copies). Returns None when absent."""
    discover_finishes()
    if primary is not None:
        return _REGISTRY.get((name, primary))
    for (n, _), f in _REGISTRY.items():
        if n == name:
            return f
    return None
