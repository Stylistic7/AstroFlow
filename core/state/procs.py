"""Processes by command line, on every platform (`pkill -f` is not on
Windows), and how this app starts its children."""
import os
import subprocess

import psutil

# A child of main.py gets its own process group on Windows: a process group
# created this way ignores the console's Ctrl-C, so a Ctrl-C — or the
# Ctrl-Break a launcher sends main.py's group — reaches main.py alone, whose
# quit then stops the children in order. Elsewhere 0, which Popen accepts.
CHILD_FLAGS = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


def matching(pattern: str) -> list[psutil.Process]:
    """Every other process whose command line contains `pattern`."""
    return [p for p in psutil.process_iter(["pid", "cmdline"])
            if p.info["pid"] != os.getpid() and pattern in " ".join(p.info["cmdline"] or ())]


def kill_matching(pattern: str, *, force: bool = False) -> int:
    """Terminate (`pkill -f`) or kill (`pkill -9 -f`) every process whose
    command line contains `pattern`; how many were signalled."""
    n = 0
    for p in matching(pattern):
        try:
            (p.kill if force else p.terminate)()
            n += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return n
