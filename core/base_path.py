import sys
import os
from pathlib import Path


def app_support_dir() -> Path:
    """Writable per-user dir for AstroFlow state that must live OUTSIDE the
    read-only / code-signed app bundle."""
    return Path.home() / "Library" / "Application Support" / "AstroFlow"


def checkout_dir() -> Path:
    """Where a worker extracts the head's committed snapshot and runs from."""
    return app_support_dir() / "checkout"


def live_core_pointer_file() -> Path:
    """A one-line file holding the path to the active checkout. Lets an
    `open`-launched bundle (which can't easily inherit env vars) find the
    checkout written by the previous run, the same way `ASTROFLOW_LIVE_CORE`
    does for a `python`-launched process."""
    return app_support_dir() / "live_core"


def _valid_base(p) -> bool:
    """A base path is usable only if it actually contains `core/` — guards
    against a stale env var or pointer file bricking startup."""
    return bool(p) and (Path(p) / "core").is_dir()


def get_base_path() -> str:
    # A synced worker runs `core/` from a writable checkout OUTSIDE the
    # read-only/code-signed bundle. The location is found via (1) the
    # ASTROFLOW_LIVE_CORE env var (python-launched re-exec), then (2) the
    # pointer file (open-launched bundle). Both point at the dir CONTAINING
    # `core/`. Each is honored only if it really holds a `core/`, so a stale
    # path falls through to the bundle/dev defaults instead of crashing.
    override = os.environ.get("ASTROFLOW_LIVE_CORE")
    if _valid_base(override):
        return override

    ptr = live_core_pointer_file()
    if ptr.exists():
        pointed = ptr.read_text().strip()
        if _valid_base(pointed):
            return pointed

    if getattr(sys, "frozen", False):
        # If the application is frozen (running as a standalone executable)
        return sys._MEIPASS
    else:
        # If the application is running in a development environment
        return Path(__file__).resolve().parent.parent


# Ray's runtime_env working_dir, when it must differ from get_base_path()/core.
# The head runs its bootstrap + web server from the full repo (which has the
# frontend/static the snapshot excludes), but points Ray task code at the
# frozen snapshot checkout so every node runs byte-identical task code.
_ray_working_dir: "str | None" = None


def set_ray_working_dir(path: str) -> None:
    global _ray_working_dir
    _ray_working_dir = path


def get_ray_working_dir() -> str:
    """The dir to ship as Ray's working_dir. Defaults to `<base>/core`; the head
    overrides it to its frozen-snapshot checkout."""
    return _ray_working_dir or f"{get_base_path()}/core"


def get_bin_dir() -> str:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "bin"
    else:
        return Path(sys.prefix) / "bin"
