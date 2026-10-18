import atexit
import os
import subprocess
import sys

import structlog

from state.state_machine import Action, enqueue_action, force_cleanup, main_event_loop
from dependency_graph.serialization import enable_function_fqname_pickling

log = structlog.get_logger()


def _prevent_idle_sleep() -> None:
    """Hold a macOS power assertion for as long as this app runs.

    A compute node that sleeps is evicted by the GCS ("consider this node to
    be dead"), and its in-flight PixInsight keeps running with no Ray task
    left to collect the result -- measured 2026-09-08/09 as ~4 hours of a
    10-core worker producing nothing, three times in one night. Worse, the
    sleep/wake cycle drops the SMB mounts, and a worker that auto-rejoins
    without the join path comes back with no shares: 438 straight
    `PermissionError: [Errno 13] /Volumes/AstroProcessed` and a node that
    fails every task in milliseconds while attracting more of them.

    macOS idle sleep keys off USER idle plus power assertions, never CPU
    load, so a node at 880% CPU mid-integration sleeps just the same. Nothing
    in this app held an assertion, so staying awake depended entirely on
    being on AC (`pmset` battery `sleep 1` vs AC `sleep 0`) -- which a
    charge-limiter's discharge cycle silently breaks.

    `-i` prevents idle sleep (not display sleep: a headless node's screen
    should still blank). `-w` makes caffeinate exit when WE exit, so there is
    nothing to clean up and no way to leak one. Best-effort by design: a
    failure here must never stop the app from booting.
    """
    if sys.platform == "win32":
        # The same for the main thread's lifetime: ES_CONTINUOUS | ES_SYSTEM_REQUIRED.
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        return
    if sys.platform != "darwin":
        return
    try:
        proc = subprocess.Popen(
            ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as e:
        log.warning("could not prevent idle sleep; this node may be evicted "
                    "if it sleeps", error=str(e))
        return
    log.info("holding a power assertion for the app's lifetime",
             caffeinate_pid=proc.pid, app_pid=os.getpid())


def main():
    atexit.register(force_cleanup)
    _prevent_idle_sleep()
    enable_function_fqname_pickling()
    enqueue_action(Action(type="init"))
    main_event_loop()


if __name__ == "__main__":
    main()
