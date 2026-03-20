"""PixInsight script execution — manages instances, slots, and IPC."""
import os
import re
import subprocess
import time
import uuid
import filelock
from contextlib import contextmanager
from pathlib import Path

import psutil

_PIXINSIGHT_EXECUTABLE = (
    "/Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight"
)
_DEFAULT_SLOTS = [1]
_LOCK_DIR = Path(os.environ.get("TMPDIR", "/tmp"))


def _find_running_instances() -> dict[int, int]:
    """Return a map of slot -> PID for running PixInsight instances."""
    try:
        result = subprocess.run(
            [_PIXINSIGHT_EXECUTABLE, "-e"],
            text=True, capture_output=True, check=True,
        )
        slot_pid = {}
        for line in result.stdout.splitlines():
            match = re.match(r"^\s*(\d+)\s+(\d+)$", line)
            if match:
                slot_pid[int(match.group(1))] = int(match.group(2))
        return slot_pid
    except Exception:
        return {}


def _start_instance(slot: int) -> None:
    """Start a PixInsight instance at the given slot, waiting until ready."""
    subprocess.Popen(
        [
            _PIXINSIGHT_EXECUTABLE,
            f"-n={slot}",
            "--no-startup-check-updates",
            "--no-startup-gui-messages",
            "--no-splash",
            "--without-printer",
        ],
        start_new_session=True,
    )
    print(f"### pix2: Started PixInsight instance at slot {slot}")

    while slot not in _find_running_instances():
        time.sleep(5)


def _kill_instance(slot: int) -> None:
    """Kill a PixInsight instance by slot number."""
    pid_map = _find_running_instances()
    pid = pid_map.get(slot)
    if pid:
        try:
            print(f"### pix2: Killing PixInsight slot {slot} (PID {pid})")
            os.kill(pid, 9)
            psutil.Process(pid).wait()
        except (psutil.NoSuchProcess, OSError):
            pass


@contextmanager
def _acquire_slot(slots: list[int]):
    """Acquire exclusive access to a PixInsight slot via file lock.

    Tries each slot in order. Blocks until one is available.
    """
    while True:
        for slot in slots:
            lock_path = _LOCK_DIR / f"pix2-slot-{slot}.lock"
            lock = filelock.FileLock(lock_path)
            try:
                lock.acquire(timeout=0)
                try:
                    yield slot
                finally:
                    lock.release()
                return
            except filelock.Timeout:
                continue

        # All slots busy — wait and retry
        time.sleep(1)


def run_javascript(
    javascript: str,
    tmp_dir: Path,
    slots: list[int] | None = None,
    timeout: float | None = None,
) -> None:
    """Execute JavaScript in a PixInsight instance.

    Acquires a slot lock, ensures an instance is running, writes the JS
    to a temp file, wraps it in a .scp script with start/end sentinels,
    and executes it via PixInsight's --execute flag.

    Args:
        javascript: The complete JavaScript to execute.
        tmp_dir: Working directory for temp files and sentinels.
        slots: PixInsight slot numbers to try. Defaults to [1].
        timeout: Maximum seconds to wait for completion.
    """
    if slots is None:
        slots = _DEFAULT_SLOTS

    with _acquire_slot(slots) as slot:
        _start_instance(slot)

        script_id = uuid.uuid4().hex[:12]
        js_filename = tmp_dir / f"{script_id}.js"
        scp_filename = tmp_dir / f"{script_id}.scp"
        start_sentinel = tmp_dir / f"start_{script_id}.txt"
        end_sentinel = tmp_dir / f"end_{script_id}.txt"

        # Write the JavaScript file
        js_filename.write_text(javascript)
        print(f"### pix2: Wrote JavaScript to {js_filename}")

        # Write the .scp shell script
        scp = (
            f'![echo "start" > {start_sentinel}\n'
            f'run -x "{js_filename}"\n'
            f'close --force *\n'
            f'![echo "end" > {end_sentinel}\n'
        )
        scp_filename.write_text(scp)
        print(f"### pix2: Wrote script to {scp_filename}")

        # Execute
        start_time = time.monotonic()
        subprocess.run(
            [_PIXINSIGHT_EXECUTABLE, f"--execute={slot}:{scp_filename}"]
        )

        # Wait for start sentinel
        while not start_sentinel.exists():
            if timeout and time.monotonic() - start_time > timeout:
                _kill_instance(slot)
                _start_instance(slot)
                raise TimeoutError(
                    f"PixInsight script did not start within {timeout}s"
                )
            time.sleep(5)

        # Wait for end sentinel
        while not end_sentinel.exists():
            if timeout and time.monotonic() - start_time > timeout:
                _kill_instance(slot)
                raise TimeoutError(
                    f"PixInsight script did not finish within {timeout}s"
                )
            time.sleep(1)

        print(f"### pix2: Script completed in {time.monotonic() - start_time:.1f}s")
