from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

_SCHEMA = "astroflow.sidecar.v2"


def sidecar_path_for(data_path: Path) -> Path:
    # One sidecar per data file, preserving the original suffix.
    return data_path.with_suffix(data_path.suffix + ".json")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_sidecar(path: Path) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError("sidecar root must be an object")
    except FileNotFoundError:
        obj = {}

    obj.setdefault("schema", _SCHEMA)
    obj.setdefault("astroflow", {})
    if not isinstance(obj["astroflow"], dict):
        raise ValueError("sidecar.astroflow must be an object")
    return obj


def save_sidecar(path: Path, obj: dict[str, Any]) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True)
    _atomic_write_text(path, text)


def _stamp_file_size(root: dict[str, Any], data_path: Path) -> None:
    """Record the data file's size so readers can detect staleness."""
    try:
        root["file_size"] = data_path.stat().st_size
    except (FileNotFoundError, OSError):
        pass


def update_sidecar(
    data_path: Path, updater: Callable[[dict[str, Any]], None]
) -> dict[str, Any]:
    sc_path = sidecar_path_for(data_path)
    root = load_sidecar(sc_path)
    updater(root)
    _stamp_file_size(root, data_path)
    save_sidecar(sc_path, root)
    return root


def get_section(data_path: Path, key: str) -> Any | None:
    sc_path = sidecar_path_for(data_path)
    root = load_sidecar(sc_path)
    return root["astroflow"].get(key)


def set_section(data_path: Path, key: str, value: Any) -> None:
    def _upd(root: dict[str, Any]) -> None:
        root["astroflow"][key] = value

    update_sidecar(data_path, _upd)


def get_file_size(data_path: Path) -> int | None:
    """Return the file_size recorded in the sidecar, or None."""
    sc_path = sidecar_path_for(data_path)
    root = load_sidecar(sc_path)
    return root.get("file_size")


