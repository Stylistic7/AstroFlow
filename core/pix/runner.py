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

# Periodic restart to mitigate PixInsight memory leaks.
_RESTART_INTERVAL = int(os.environ.get("ASTROFLOW_PIX_RESTART_INTERVAL", "20"))
_MAX_MEMORY_GB = float(os.environ.get("ASTROFLOW_PIX_MAX_MEMORY_GB", "32"))


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
    """Ensure a PixInsight instance is running at the given slot.

    If an instance is already running (and its PID is alive), this is a no-op.
    Otherwise, waits for any stale registration to clear, then launches a new
    instance and waits until it registers.
    """
    existing = _find_running_instances()
    if slot in existing:
        pid = existing[slot]
        if psutil.pid_exists(pid):
            return  # Already running
        # Stale registration — wait for it to clear
        print(f"### pix: Slot {slot} has stale PID {pid}, waiting for cleanup")
        while slot in _find_running_instances():
            time.sleep(2)

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
    print(f"### pix: Started PixInsight instance at slot {slot}")

    while slot not in _find_running_instances():
        time.sleep(5)


def _counter_path(slot: int) -> Path:
    return _LOCK_DIR / f"pix-slot-{slot}.count"


def _should_restart(slot: int) -> bool:
    """Check if PixInsight should be restarted (task count or memory).

    Called while the slot lock is held — counter file access is race-free.
    """
    # Increment task counter.
    path = _counter_path(slot)
    count = int(path.read_text()) if path.exists() else 0
    count += 1
    path.write_text(str(count))

    if _RESTART_INTERVAL > 0 and count >= _RESTART_INTERVAL:
        print(f"### pix: Slot {slot} hit {count} tasks, restarting")
        path.write_text("0")
        return True

    # Check memory.
    if _MAX_MEMORY_GB > 0:
        pid = _find_running_instances().get(slot)
        if pid:
            try:
                mem_gb = psutil.Process(pid).memory_info().rss / (1024**3)
                if mem_gb > _MAX_MEMORY_GB:
                    print(f"### pix: Slot {slot} using {mem_gb:.1f} GB (limit {_MAX_MEMORY_GB}), restarting")
                    path.write_text("0")
                    return True
            except psutil.NoSuchProcess:
                pass

    return False


def _kill_instance(slot: int) -> None:
    """Shut down a PixInsight instance by slot number.

    First tries a graceful exit via PixInsight's own quit command, giving
    it time to clean up its slot registration. Falls back to SIGKILL if
    the graceful shutdown doesn't complete within 30 seconds.
    """
    pid_map = _find_running_instances()
    pid = pid_map.get(slot)
    if not pid:
        return

    print(f"### pix: Requesting graceful exit for slot {slot} (PID {pid})")
    exit_script = _LOCK_DIR / f"pix-exit-{slot}.scp"
    exit_script.write_text("exit\n")
    subprocess.run(
        [_PIXINSIGHT_EXECUTABLE, f"--execute={slot}:{exit_script}"],
        timeout=10,
    )

    # Wait for the process to exit
    try:
        proc = psutil.Process(pid)
        proc.wait(timeout=30)
        print(f"### pix: Slot {slot} exited gracefully")
    except psutil.TimeoutExpired:
        print(f"### pix: Slot {slot} did not exit gracefully, sending SIGKILL")
        try:
            os.kill(pid, 9)
            psutil.Process(pid).wait(timeout=10)
        except (psutil.NoSuchProcess, OSError, psutil.TimeoutExpired):
            pass
    except psutil.NoSuchProcess:
        pass  # Already gone


@contextmanager
def _acquire_slot(slots: list[int]):
    """Acquire exclusive access to a PixInsight slot via file lock.

    Tries each slot in order. Blocks until one is available.
    """
    while True:
        for slot in slots:
            lock_path = _LOCK_DIR / f"pix-slot-{slot}.lock"
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


def run_scp(
    scp_commands: str,
    tmp_dir: Path,
    slots: list[int] | None = None,
    timeout: float | None = None,
) -> None:
    """Execute a .scp script in a PixInsight instance.

    Wraps the given commands with start/end sentinels and a close-all,
    then executes via PixInsight's --execute flag.

    Args:
        scp_commands: The .scp commands (run -x lines, etc.)
        tmp_dir: Working directory for temp files and sentinels.
        slots: PixInsight slot numbers to try. Defaults to [1].
        timeout: Maximum seconds to wait for completion.
    """
    if slots is None:
        slots = _DEFAULT_SLOTS

    with _acquire_slot(slots) as slot:
        _start_instance(slot)

        script_id = uuid.uuid4().hex[:12]
        scp_filename = tmp_dir / f"{script_id}.scp"
        start_sentinel = tmp_dir / f"start_{script_id}.txt"
        end_sentinel = tmp_dir / f"end_{script_id}.txt"

        # Wrap commands with sentinels and close-all
        scp = (
            f'![echo "start" > {start_sentinel}\n'
            f'{scp_commands}\n'
            f'close --force *\n'
            f'![echo "end" > {end_sentinel}\n'
        )
        scp_filename.write_text(scp)
        print(f"### pix: Wrote script to {scp_filename}")

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

        print(f"### pix: Script completed in {time.monotonic() - start_time:.1f}s")

        if _should_restart(slot):
            _kill_instance(slot)
