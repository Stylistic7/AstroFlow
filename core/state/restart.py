"""Process self-restart for pull-deploy.

A restart = spawn a detached replacement, then let the existing graceful `quit`
action tear down zeroconf / Ray / gunicorn / mounts and exit. We deliberately
do NOT `os.execv` into the app bundle: the bundle's `core/` is read-only and
code-signed, so execv there would reload the *bundled* code, not the synced
checkout. Spawn-and-quit also reuses the shutdown path that already works.

The caller (a state-machine handler) returns `Action("quit")` after invoking
`spawn_replacement` — keeping this module free of any state-machine import.
"""
import os
import subprocess
import sys
from pathlib import Path

from base_path import get_base_path, live_core_pointer_file


def _bundle_path() -> Path:
    # sys.executable = AstroFlow.app/Contents/MacOS/AstroFlow → the .app is 3 up.
    return Path(sys.executable).resolve().parents[2]


def spawn_replacement(live_core: str | None = None) -> None:
    """Launch a fresh AstroFlow process. If `live_core` is given (a worker that
    just synced new code), the replacement runs `core/` from that checkout."""
    env = dict(os.environ)

    if live_core:
        env["ASTROFLOW_LIVE_CORE"] = str(live_core)
        # `open` doesn't propagate env to a bundle, so also drop a pointer file
        # that base_path.get_base_path() reads.
        ptr = live_core_pointer_file()
        ptr.parent.mkdir(parents=True, exist_ok=True)
        ptr.write_text(str(live_core))

    if getattr(sys, "frozen", False):
        # Relaunch the bundle; `open -n` lets LaunchServices manage activation.
        subprocess.Popen(
            ["open", "-n", str(_bundle_path())],
            env=env,
            start_new_session=True,
        )
    else:
        # Dev: re-run main.py from the target checkout (or our own repo).
        base = live_core or get_base_path()
        main_py = Path(base) / "core" / "main.py"
        subprocess.Popen(
            [sys.executable, str(main_py)],
            env=env,
            start_new_session=True,
        )
