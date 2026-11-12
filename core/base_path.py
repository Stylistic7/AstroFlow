import sys
import os
from pathlib import Path


def app_support_dir() -> Path:
    """Writable per-user dir for AstroFlow state that must live OUTSIDE the
    read-only / code-signed app bundle."""
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "AstroFlow"
    return Path.home() / "Library" / "Application Support" / "AstroFlow"


def ray_temp_root() -> str:
    """Where Ray keeps its session (`<root>/ray/session_*`: logs, the Unix
    sockets, and the copies of `core/` it ships to every worker). Not `/tmp`,
    Ray's default on macOS: `/usr/libexec/tmp_cleaner` runs at midnight and
    deletes any file there untouched for three days, and on 2026-11-12 it
    emptied the shipped `core/` copies of a head four days up — every worker
    spawned after that failed its first import, every pass ended with nothing
    it could run, and every object with new frames read OUT OF DATE. `/var/tmp`
    is not swept. Short, because the sockets' paths
    (`…/session_<32 chars>/sockets/plasma_store`) must fit macOS's 104 bytes."""
    if sys.platform == "win32":
        return str(app_support_dir() / "tmp")
    return "/var/tmp/astroflow"


def get_base_path() -> str:
    # The app runs the checkout (or app bundle copy) it was launched from —
    # full stop. The old ASTROFLOW_LIVE_CORE / pointer-file redirect existed
    # so a synced worker could re-exec into a pulled snapshot; with code
    # moving via git, a redirect is only a way to silently run the WRONG tree
    # (a stale pointer on a worker pointed at a weeks-old frozen checkout,
    # 2026-08-25).
    return Path(__file__).resolve().parent.parent


# Ray's runtime_env working_dir. Set explicitly at head startup to
# `<base>/core` (the live checkout) — same tree the version fence describes.
_ray_working_dir: "str | None" = None


def set_ray_working_dir(path: str) -> None:
    global _ray_working_dir
    _ray_working_dir = path


def get_ray_working_dir() -> str:
    """The dir to ship as Ray's working_dir: `<base>/core`."""
    return _ray_working_dir or f"{get_base_path()}/core"


# The Ray CLI, run by this interpreter. Not the `ray` console script: its
# shebang pins the interpreter's absolute path, which breaks once the macOS
# app bundle is moved, and an unquoted path with a space broke the shell
# commands that launch it.
RAY_CLI = (sys.executable, "-m", "ray.scripts.scripts")
