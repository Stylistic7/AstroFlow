"""`state.procs`: processes by command line, the way `pkill -f` found them."""
import subprocess
import sys
import uuid

from state import procs


def _sleeper(marker: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", marker],
                            creationflags=procs.CHILD_FLAGS)


def test_kill_matching_signals_every_process_naming_the_pattern_but_not_this_one():
    marker = f"astroflow-test-{uuid.uuid4().hex}"
    children = [_sleeper(marker) for _ in range(2)]
    try:
        assert sorted(p.pid for p in procs.matching(marker)) == sorted(c.pid for c in children)
        assert procs.kill_matching(marker) == 2
        for child in children:
            child.wait(timeout=10)
            assert child.returncode != 0, "terminated, not exited on its own"
    finally:
        for child in children:
            child.kill()


def test_a_pattern_nothing_runs_signals_nothing():
    assert procs.kill_matching(f"astroflow-test-{uuid.uuid4().hex}") == 0
