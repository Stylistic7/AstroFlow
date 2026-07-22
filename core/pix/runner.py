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


def _slot_registered(slot: int) -> bool:
    """Strict check: PI's own IPC reports slot `slot` as registered.

    Unlike `_find_running_instances`, this does NOT fall back to psutil —
    it's purely a "PI is ready to receive --execute={slot}:" probe. PI
    process existence (psutil) becomes true the instant Popen returns,
    well before PI has set up its Mach IPC, so it's not a valid readiness
    signal. Only `-e` returning the slot is.
    """
    try:
        result = subprocess.run(
            [_PIXINSIGHT_EXECUTABLE, "-e"],
            text=True, capture_output=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        return False
    if result.returncode != 0:
        return False
    return slot in _parse_e_output(result.stdout)


def _start_instance(slot: int) -> None:
    """Ensure a PixInsight instance is running at `slot` and is ready to
    accept --execute= dispatches.

    Readiness means PI's own IPC reports the slot — not just that the
    Popen'd PID exists. PID-existence is true within milliseconds of
    Popen, but PI takes several seconds (longer on slower hosts) to
    finish initialising Qt and register its slot in the Mach IPC table.
    Returning early lets the next `subprocess.run([PI, --execute=N:])`
    block waiting for slot N to become reachable, which on the MacBook
    Pro burned the full start-sentinel timeout (60s) every restart and
    triggered cascading "Start sentinel not written" restart loops.
    """
    existing = _find_running_instances()
    if slot in existing:
        pid = existing[slot]
        if psutil.pid_exists(pid):
            return  # Already running and registered
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

    # Wait until PI's IPC reports the slot. Each iteration costs one
    # `PixInsight -e` subprocess (~1-10s); during boot `-e` typically
    # fails fast with returncode != 0, so iterations are cheap until PI
    # is actually ready.
    started_at = time.monotonic()
    deadline = started_at + 120
    while not _slot_registered(slot):
        if not psutil.pid_exists(proc.pid):
            raise RuntimeError(
                f"PixInsight pid {proc.pid} died before registering slot {slot}"
            )
        if time.monotonic() > deadline:
            raise TimeoutError(f"PixInsight slot {slot} did not register within 120s")
        time.sleep(2)
    print(f"### pix: Slot {slot} ready after {time.monotonic() - started_at:.1f}s")


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


# Overall ceiling on waiting for a free PixInsight slot. The slot file
# lock sits behind Ray's per-node `PixInsight` resource, which already
# gates concurrency to one task per slot — so a wait this long means a
# stale lock or drifted resource accounting, NOT legitimate contention.
# Raising converts an unrecoverable spin (a worker holding its Ray
# `PixInsight` slot forever at 0% CPU) into a task failure that Ray
# retries and reschedules via the @node's `max_retries`.
_SLOT_ACQUIRE_TIMEOUT = float(
    os.environ.get("ASTROFLOW_PIX_SLOT_ACQUIRE_TIMEOUT", "300")
)


@contextmanager
def _acquire_slot(slots: list[int]):
    """Acquire exclusive access to a PixInsight slot via file lock.

    Tries each slot in order. Waits up to `_SLOT_ACQUIRE_TIMEOUT` for one
    to free, then raises `TimeoutError` rather than blocking forever.
    """
    deadline = time.monotonic() + _SLOT_ACQUIRE_TIMEOUT
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

        # All slots busy — fail fast once the ceiling is hit so the task
        # surfaces as a retryable error instead of deadlocking the slot.
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Could not acquire a PixInsight slot from {slots} within "
                f"{_SLOT_ACQUIRE_TIMEOUT:.0f}s — stale lock or stuck slot"
            )
        time.sleep(1)


_START_SENTINEL_TIMEOUT = 60      # seconds to wait for PixInsight to begin executing

# Liveness watchdog for the running script. A flat wall-clock cap is wrong
# for this workload: a big WBPP integration legitimately takes hours, so a
# fixed cap guillotines it mid-run and throws the work away — while a PI
# wedged on a modal error dialog would burn the whole cap doing nothing. So
# instead of capping total runtime we cap INACTIVITY: keep waiting as long
# as the instance shows signs of life, and kill only after
# `_PIX_INACTIVITY_TIMEOUT` seconds with no progress. "Progress" =
#   * the OS reporting the process burned CPU (`cpu_times`, the PRIMARY
#     signal — true even during a silent final integration that writes
#     nothing until the end), OR
#   * new bytes landing in the watch dir (WBPP's per-frame intermediates).
# A stuck-on-a-dialog PI is quiet on both and gets reaped in minutes.
# (psutil exposes no per-process `io_counters()` on macOS, so watch-dir
# mtime is the I/O proxy; CPU carries the rest.)
_PIX_INACTIVITY_TIMEOUT = float(
    os.environ.get("ASTROFLOW_PIX_INACTIVITY_TIMEOUT", "300")
)
_CPU_ACTIVE_FRACTION = 0.05       # ≥5% of one core over a poll interval = "working"
_WATCHDOG_POLL_INTERVAL = 2.0     # seconds between liveness samples
# `--execute` only DISPATCHES the script to the persistent slot instance
# and returns promptly (actual execution is tracked via the sentinels
# below) — so this guards the dispatch client itself wedging, not the
# script runtime. Without it the unguarded dispatch could hang a worker
# forever, holding its Ray `PixInsight` slot at 0% CPU.
_DISPATCH_TIMEOUT = float(os.environ.get("ASTROFLOW_PIX_DISPATCH_TIMEOUT", "60"))


def run_scp(
    scp_commands: str,
    tmp_dir: Path,
    slots: list[int] | None = None,
    watch_dir: Path | None = None,
    timeout: float | None = None,
) -> None:
    """Execute a .scp script in a PixInsight instance.

    Wraps the given commands with start/end sentinels and a close-all,
    then executes via PixInsight's --execute flag.

    Args:
        scp_commands: The .scp commands (run -x lines, etc.)
        tmp_dir: Working directory for temp files and sentinels.
        slots: PixInsight slot numbers to try. Defaults to [1].
        watch_dir: Directory whose file-write activity counts as progress
            for the inactivity watchdog. Defaults to `tmp_dir` (where a
            recipe writes its output); WBPP passes its separate
            `output_dir`, since that — not the sentinel `tmp_dir` — is where
            it streams per-frame intermediates.
        timeout: Optional ABSOLUTE wall-clock ceiling in seconds; None
            (default) = unlimited. The real liveness bound is the inactivity
            watchdog (`_PIX_INACTIVITY_TIMEOUT`); this is only a hard cap for
            callers that want one (e.g. tests). A legitimately-long stack
            passes None and runs as long as it keeps making progress.
    """
    if slots is None:
        slots = _DEFAULT_SLOTS

    with _acquire_slot(slots) as slot:
        _start_instance(slot)

        script_id = uuid.uuid4().hex[:12]
        scp_filename = tmp_dir / f"{script_id}.scp"
        start_sentinel = tmp_dir / f"start_{script_id}.txt"
        end_sentinel = tmp_dir / f"end_{script_id}.txt"

        # Wrap commands with sentinels and close-all. The leading
        # `close --force *` makes every recipe self-cleaning on entry:
        # a prior recipe in this persistent instance (especially one that
        # failed) may have left views open, and reopening the same file
        # yields the same deterministic view_id — so start from a clean
        # slate to avoid collisions. Trailing close-all tidies up after.
        scp = (
            f'![echo "start" > {start_sentinel}\n'
            f'close --force *\n'
            f'{scp_commands}\n'
            f'close --force *\n'
            f'![echo "end" > {end_sentinel}\n'
        )
        scp_filename.write_text(scp)
        print(f"### pix: Wrote script to {scp_filename}")

        # Execute. The dispatch returns promptly; the start/end sentinels
        # below track the real execution. `timeout` guards the dispatch
        # client wedging — on timeout subprocess.run kills it, then we
        # restart the slot instance and raise so Ray retries the task
        # rather than the worker deadlocking on a held slot.
        start_time = time.monotonic()
        try:
            subprocess.run(
                [_PIXINSIGHT_EXECUTABLE, f"--execute={slot}:{scp_filename}"],
                timeout=_DISPATCH_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            print(
                f"### pix: --execute dispatch wedged >{_DISPATCH_TIMEOUT:.0f}s "
                f"on slot {slot}, killing instance"
            )
            _kill_instance(slot)
            raise TimeoutError(
                f"PixInsight --execute dispatch did not return within "
                f"{_DISPATCH_TIMEOUT:.0f}s on slot {slot}"
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

        # Wait for the end sentinel — the real script execution. Rather than
        # a flat wall-clock cap (which kills a legitimately-hours-long WBPP
        # integration and wastes it), watch for LIVENESS: keep waiting as
        # long as PixInsight burns CPU (kernel-reported `cpu_times` — true
        # even during a silent final integration that writes nothing until
        # the end) or writes new bytes into `watch_dir` (WBPP's per-frame
        # intermediates), and kill only after `_PIX_INACTIVITY_TIMEOUT`
        # seconds with NO progress — what a PI stuck on a modal error dialog
        # looks like. The per-poll alive check still surfaces a crash /
        # user-quit within seconds; `timeout`, if given, is an extra absolute
        # ceiling on top of the watchdog.
        watch_dir = watch_dir or tmp_dir
        proc = psutil.Process(instance_pid) if instance_pid else None
        absolute_deadline = start_time + timeout if timeout is not None else None

        def _cpu_seconds() -> float:
            if proc is None:
                return 0.0
            try:
                t = proc.cpu_times()
            except psutil.NoSuchProcess:
                return 0.0
            return t.user + t.system

        def _latest_write_mtime() -> float:
            try:
                return max((p.stat().st_mtime for p in watch_dir.rglob("*")
                            if p.is_file()), default=0.0)
            except OSError:
                return 0.0

        last_progress = prev_sample = time.monotonic()
        prev_cpu, prev_mtime = _cpu_seconds(), _latest_write_mtime()
        while not end_sentinel.exists():
            if not _instance_alive():
                print(f"### pix: PixInsight died during script execution")
                _start_instance(slot)
                raise TimeoutError("PixInsight exited during script execution")

            now = time.monotonic()
            cpu, mtime = _cpu_seconds(), _latest_write_mtime()
            # "Working" = burned >5% of one core since the last sample (any
            # real work, single- or multi-threaded, clears this by 20×+), OR
            # a new/updated file landed in the watch dir.
            active = (cpu - prev_cpu > _CPU_ACTIVE_FRACTION * (now - prev_sample)
                      or mtime > prev_mtime)
            prev_cpu, prev_mtime, prev_sample = cpu, mtime, now
            if active:
                last_progress = now
            elif now - last_progress > _PIX_INACTIVITY_TIMEOUT:
                idle = now - last_progress
                print(
                    f"### pix: No CPU or output activity for {idle:.0f}s "
                    f"(inactivity limit {_PIX_INACTIVITY_TIMEOUT:.0f}s), "
                    f"PixInsight presumed stuck — killing instance"
                )
                _kill_instance(slot)
                raise TimeoutError(f"PixInsight made no progress for {idle:.0f}s (stuck?)")

            if absolute_deadline is not None and now > absolute_deadline:
                print(f"### pix: Exceeded absolute timeout of {timeout:.0f}s, killing instance")
                _kill_instance(slot)
                raise TimeoutError(f"PixInsight script did not finish within {timeout:.0f}s")
            time.sleep(_WATCHDOG_POLL_INTERVAL)

        print(f"### pix: Script completed in {time.monotonic() - start_time:.1f}s")

        if _should_restart(slot):
            _kill_instance(slot)
