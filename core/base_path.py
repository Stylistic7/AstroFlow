import sys
import os
from pathlib import Path


def app_support_dir() -> Path:
    """Writable per-user dir for AstroFlow state that must live OUTSIDE the
    read-only / code-signed app bundle."""
    return Path.home() / "Library" / "Application Support" / "AstroFlow"


def get_base_path() -> str:
    # The app runs the checkout it was launched from — full stop. The old
    # ASTROFLOW_LIVE_CORE / pointer-file redirect existed so a synced worker
    # could re-exec into a pulled snapshot; with code moving via git, a
    # redirect is only a way to silently run the WRONG tree (a stale pointer
    # on a worker pointed at a weeks-old frozen checkout, 2026-08-25).
    if getattr(sys, "frozen", False):
        # If the application is frozen (running as a standalone executable)
        return sys._MEIPASS
    else:
        # If the application is running in a development environment
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


def get_bin_dir() -> str:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "bin"
    else:
        return Path(sys.prefix) / "bin"
