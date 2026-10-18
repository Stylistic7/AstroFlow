"""The user's catalogue and tool settings, as every Mac in the cluster reads them.

Code that runs on any node (a @node body solving a frame, a task calling
GraXpert) can't read the durable store, so the head copies the user's choices
(the Third-Party Tools page) into a small file on each Mac: when the
head starts, whenever they are saved (a task pinned to every node), and on a
worker when it joins (`state_machine._join_head`). Readers open the file when
they need a value, so a change applies to the next call.

  {"astrometry_index_dir": "/Volumes/…/astrometry-net" | null,
   "tool_paths": {"graxpert": "/opt/…/graxpert", …}}

`astrometry_index_dir` is the user's EXTRA index files, for fields smaller
than the app's own set covers (about 20'); from a checkout it is the only one.

Beside it, solve-field's own configuration naming the index folders
(`astrometry.cfg`): the app's (`ASTROFLOW_INDEX_DIR`, set by the app on each
Mac) and the extra one. `plate_solving` passes it whenever it exists.
"""
import json
import os
from pathlib import Path

from base_path import app_support_dir

SUPPORT = app_support_dir()
SETTINGS = SUPPORT / "settings.json"
ASTROMETRY_CONFIG = SUPPORT / "astrometry.cfg"


def read() -> dict:
    try:
        return json.loads(SETTINGS.read_text())
    except FileNotFoundError:
        return {}


def write(settings: dict) -> None:
    SUPPORT.mkdir(parents=True, exist_ok=True)
    _replace(SETTINGS, json.dumps(settings, indent=2))
    folders = index_folders(settings)
    if folders:
        paths = "".join(f"add_path {f}\n" for f in folders)
        _replace(ASTROMETRY_CONFIG, f"inparallel\ncpulimit 300\n{paths}autoindex\n")
    else:
        ASTROMETRY_CONFIG.unlink(missing_ok=True)


def index_folders(settings: dict) -> list[str]:
    """The folders solve-field reads index files from on this Mac: the app's
    own, then the user's extra one."""
    bundled = os.environ.get("ASTROFLOW_INDEX_DIR")
    return [f for f in (bundled, settings.get("astrometry_index_dir")) if f]


def _replace(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    tmp.write_text(text)
    tmp.replace(path)


def current() -> dict:
    """The settings as the head's durable store holds them."""
    from context import local_context as context
    index = context.astrometry_index_dir
    return {
        "astrometry_index_dir": str(index) if index else None,
        "tool_paths": dict(context.tool_paths or {}),
    }


def push_to_cluster(settings: dict) -> None:
    """Writes `settings` on every Mac in the cluster. A Mac that doesn't
    take them aborts the request (503), so the caller saves nothing."""
    from flask import abort
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from ray_infra.ray_init import connect_to_ray_cluster
    connect_to_ray_cluster()  # a web worker connects on first use
    write_remote = ray.remote(num_cpus=0)(write)
    refs = [write_remote.options(scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False))
            .remote(settings) for n in ray.nodes() if n.get("Alive")]
    try:
        ray.get(refs, timeout=60)
    except Exception as e:
        abort(503, description=f"Not saved: a Mac in the cluster didn't take the settings ({e})")
