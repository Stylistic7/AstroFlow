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


def _parse_e_output(stdout: str) -> dict[int, int]:
    slot_pid: dict[int, int] = {}
    for line in stdout.splitlines():
        match = re.match(r"^\s*(\d+)\s+(\d+)$", line)
        if match:
            slot_pid[int(match.group(1))] = int(match.group(2))
    return slot_pid


def _find_via_psutil() -> dict[int, int]:
    """OS-level fallback: scan the process table for PixInsight instances
    we launched with `-n=<slot>`. Less authoritative than PI's own
    registry — it can't see slots that are registered but whose process
    has died, and won't find a manually-launched PI without `-n=`. Good
    enough to prevent duplicate spawns when `-e` is broken.
    """
    instances: dict[int, int] = {}
    for proc in psutil.process_iter(["pid", "exe", "cmdline"]):
        try:
            if proc.info["exe"] != _PIXINSIGHT_EXECUTABLE:
                continue
            for arg in proc.info["cmdline"] or []:
                if arg.startswith("-n="):
                    instances[int(arg.removeprefix("-n="))] = proc.info["pid"]
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return instances


def _find_running_instances() -> dict[int, int]:
    """Return a map of slot -> PID for running PixInsight instances.

    Authoritative path: ask PI itself via `-e`. Fallback: scan the OS
    process table when `-e` fails. We must NOT silently treat "couldn't
    query PI" as "registry empty" — that conflation is what causes
    duplicate spawns to cascade when PI's IPC is broken on a host.
    """
    try:
        result = subprocess.run(
            [_PIXINSIGHT_EXECUTABLE, "-e"],
            text=True, capture_output=True, timeout=30,
        )
    except subprocess.TimeoutExpired as e:
        print(f"### pix: -e timed out after 30s; falling back to psutil. {e}")
        return _find_via_psutil()
    except FileNotFoundError as e:
        print(f"### pix: PixInsight executable missing at {_PIXINSIGHT_EXECUTABLE}: {e}")
        raise

    if result.returncode != 0:
        print(
            f"### pix: -e exited {result.returncode}; "
            f"stderr={result.stderr.strip()!r}; falling back to psutil"
        )
        return _find_via_psutil()

    return _parse_e_output(result.stdout)


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
        # Stale registration — wait for it to clear (max 60s)
        print(f"### pix: Slot {slot} has stale PID {pid}, waiting for cleanup")
        deadline = time.monotonic() + 60
        while slot in _find_running_instances():
            if time.monotonic() > deadline:
                raise TimeoutError(f"Stale PixInsight slot {slot} did not clear within 60s")
            time.sleep(2)

    proc = subprocess.Popen(
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
    print(f"### pix: Started PixInsight instance at slot {slot} (pid {proc.pid})")

    # Wait for the spawned process to be alive. We have its PID directly
    # from Popen, so a `pid_exists` check is enough — no need to ask PI's
    # own registry, which would cost a subprocess invocation per poll
    # (and a dock-icon flash on macOS) for every iteration.
    deadline = time.monotonic() + 120
    while not psutil.pid_exists(proc.pid):
        if time.monotonic() > deadline:
            raise TimeoutError(f"PixInsight slot {slot} did not register within 120s")
        time.sleep(1)


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


_START_SENTINEL_TIMEOUT = 60      # seconds to wait for PixInsight to begin executing
_END_SENTINEL_TIMEOUT = 3600      # seconds to wait for the script to finish (1 hour)


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
        timeout: Maximum seconds to wait for the end sentinel.
            Defaults to _END_SENTINEL_TIMEOUT (1 hour).
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

        instance_pid = _find_running_instances().get(slot)

        def _instance_alive() -> bool:
            return instance_pid is not None and psutil.pid_exists(instance_pid)

        # Wait for start sentinel — if PixInsight crashed, was quit by
        # the user, or never began executing the script, detect it
        # immediately via the process check rather than waiting for a
        # timeout.
        start_deadline = start_time + _START_SENTINEL_TIMEOUT
        while not start_sentinel.exists():
            if not _instance_alive():
                print(f"### pix: PixInsight died before writing start sentinel, restarting")
                _start_instance(slot)
                raise TimeoutError("PixInsight exited before script started")
            if time.monotonic() > start_deadline:
                print(f"### pix: Start sentinel not written after {_START_SENTINEL_TIMEOUT}s, restarting instance")
                _kill_instance(slot)
                _start_instance(slot)
                raise TimeoutError(
                    f"PixInsight script did not start within {_START_SENTINEL_TIMEOUT}s"
                )
            time.sleep(5)

        # Wait for end sentinel — the actual script execution. WBPP on
        # a big stack can legitimately take an hour. Check that
        # PixInsight is still alive on every poll so a user-quit is
        # detected within seconds, not after the full timeout.
        end_deadline = start_time + (timeout if timeout is not None else _END_SENTINEL_TIMEOUT)
        while not end_sentinel.exists():
            if not _instance_alive():
                print(f"### pix: PixInsight died during script execution")
                _start_instance(slot)
                raise TimeoutError("PixInsight exited during script execution")
            if time.monotonic() > end_deadline:
                print(f"### pix: End sentinel not written after timeout, killing instance")
                _kill_instance(slot)
                raise TimeoutError("PixInsight script did not finish within timeout")
            time.sleep(1)

        print(f"### pix: Script completed in {time.monotonic() - start_time:.1f}s")

        if _should_restart(slot):
            _kill_instance(slot)
