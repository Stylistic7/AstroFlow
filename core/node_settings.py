"""The user's catalogue and tool settings, as every Mac in the cluster reads them.

Code that runs on any node (a @node body solving a frame, a task calling
GraXpert) can't read the durable store, so the head copies the user's choices
(Catalogs and Third-Party Tools pages) into a small file on each Mac: when the
head starts, whenever they are saved (a task pinned to every node), and on a
worker when it joins (`state_machine._join_head`). Readers open the file when
they need a value, so a change applies to the next call.

  {"astrometry_index_dir": "/Volumes/…/astrometry-net" | null,
   "mars_database": "/Volumes/…/MARS-DR1-1.1.1.xmars" | null,
   "tool_paths": {"graxpert": "/opt/…/graxpert", …}}

Beside it, solve-field's own configuration naming the index folder
(`astrometry.cfg`); `plate_solving` passes it whenever it exists.
"""
import json
import os
from pathlib import Path

SUPPORT = Path.home() / "Library" / "Application Support" / "AstroFlow"
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
    index = settings.get("astrometry_index_dir")
    if index:
        _replace(ASTROMETRY_CONFIG, f"inparallel\ncpulimit 300\nadd_path {index}\nautoindex\n")
    else:
        ASTROMETRY_CONFIG.unlink(missing_ok=True)


def _replace(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    tmp.write_text(text)
    tmp.replace(path)


def current() -> dict:
    """The settings as the head's durable store holds them."""
    from context import local_context as context
    index, mars = context.astrometry_index_dir, context.mars_database
    return {
        "astrometry_index_dir": str(index) if index else None,
        "mars_database": str(mars) if mars else None,
        "tool_paths": dict(context.tool_paths or {}),
    }


def push_to_cluster() -> None:
    """Writes the head's settings on every Mac in the cluster."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    settings = current()
    write_remote = ray.remote(num_cpus=0)(write)
    refs = [write_remote.options(scheduling_strategy=NodeAffinitySchedulingStrategy(n["NodeID"], soft=False))
            .remote(settings) for n in ray.nodes() if n.get("Alive")]
    ray.get(refs, timeout=60)
