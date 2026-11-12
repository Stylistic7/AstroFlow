import os
import shlex
import shutil
import subprocess
import sys
import time
import socket
from pathlib import Path

import psutil

from base_path import RAY_CLI, ray_temp_root
from state.procs import CHILD_FLAGS

# Every `ray` command here and `ray.init` (ray_infra.ray_init) must agree on
# the session directory: `ray start` creates it there, `ray status` and
# `address="auto"` read `<RAY_TMPDIR>/ray/ray_current_cluster`.
RAY_ENV = {"RAY_TMPDIR": ray_temp_root()}


def _workers_soft_limit() -> int:
    """Idle-worker soft limit: a quarter of the local box's cores.

    Left unset, Ray's soft limit defaults to num_cpus, so it keeps ~num_cpus
    idle workers resident (and prestarts one per core at startup). On the 24-core
    head that's ~24 idle Python workers holding ~15 GB even when nothing is
    running — which drops the head's free memory below the worker's, so the
    memory-aware scheduler (`_select_node` places on the ROOMIEST node) steers
    heavy ops onto the worker and the head sits idle. It is also pure waste when
    an external heavyweight process (PixInsight, a stack) owns the whole box.

    Exported as RAY_num_workers_soft_limit at `ray start`, this caps the warm
    pool low. num_cpus stays on auto, so Ray still scales the pool UP to full
    cores on demand for light-op bursts and reaps back down to this floor when
    idle — no task-concurrency is surrendered. Per-machine, so nodes self-size.
    """
    return max(1, (os.cpu_count() or 4) // 4)


def _ray_memory_bytes() -> int:
    """Memory to advertise to Ray as the node's static `memory` resource:
    all of physical RAM.

    Ray's default at startup is `available × 0.7`, taken once — so if
    PixInsight (or anything else heavy) was running when the head started,
    the cluster ends up with a permanently smaller cap than the machine
    can actually deliver; anchoring on `total` gives a stable number. It is
    the whole of RAM, not 0.7 of it, because reservations are p95 PEAKS held
    for a task's whole run (and Siril's is a fixed budget): at 0.7 the ledger
    filled at ~3x the memory actually in use and held Siril with 59 % of the
    Studio free (2026-11-06). The real brakes are `_select_node`'s measured
    free memory and the macOS pressure veto.
    """
    return int(psutil.virtual_memory().total)


def is_ray_connected(timeout: float = 10.0):
    """Whether the local Ray node is up (a local start/stop lifecycle barrier).

    Used ONLY by `_wait_for_ray` (poll until `ray start` has bound the node) and
    `stop_ray_node` (poll until `ray stop` has torn it down). `ray status`
    targets whatever cluster the local session last pointed at, so we cap it with
    a timeout and report "not connected" on timeout rather than blocking. We do
    NOT use this to police a running connection — Ray owns health/connection
    tracking, and probing it on top and acting on a flaky result tore down
    healthy raylets.
    """
    try:
        result = subprocess.run(
            [*RAY_CLI, "status"], capture_output=True, text=True, timeout=timeout,
            env={**os.environ, **RAY_ENV},
        )
    except subprocess.TimeoutExpired:
        return False
    return "Autoscaler status" in result.stdout


def _fresh_temp_dir() -> None:
    """Clear Ray's session directory before this node starts: nothing else
    ever does (macOS swept `/tmp/ray`; `ray_temp_root` is out of its reach),
    and a session is logs plus the shipped code — gigabytes a boot. The
    newest session stays as the post-mortem of whatever ended the last run.
    A live local node (`ray status` answers) keeps everything: its sockets
    are in there."""
    root = Path(ray_temp_root()) / "ray"
    if not root.is_dir() or is_ray_connected():
        return
    sessions = sorted(p for p in root.glob("session_*") if p.is_dir() and not p.is_symlink())
    for stale in sessions[:-1]:
        shutil.rmtree(stale, ignore_errors=True)


def _wait_for_ray(process: subprocess.Popen, timeout):
    """Poll `ray status` until the node is up; raise as soon as `ray start`
    exits with an error (it exits 0 once the node runs), or after `timeout`."""
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if process.poll() not in (None, 0):
            raise RuntimeError(f"`ray start` exited with {process.returncode}; its output is above")
        print(f"### Waiting for Ray node to start... ({time.monotonic() - started:.0f}s)")
        if is_ray_connected():
            return
        time.sleep(1)
    raise TimeoutError(f"Ray node did not start within {timeout}s")


def _resources(base: dict) -> str:
    """The node's custom Ray resources as `ray start --resources` wants them.
    `Siril: 1` marks a node with Siril installed: integrations request a
    sliver of it, so it gates where they run, not how many."""
    import json
    from third_party_tools import ToolNotFoundError, tool_path
    try:
        if os.path.exists(tool_path("siril")):
            base = {**base, "Siril": 1}
    except ToolNotFoundError:
        pass
    return json.dumps(base)


def _start_ray(args: list[str], env: dict[str, str]) -> subprocess.Popen:
    """`ray start` as an argument list with its own environment: no shell,
    which Windows lacks `ulimit` and `VAR=value cmd` for. The open-file limit
    the raylet needs (10,000) is raised here, in this process, and inherited;
    Windows has no such limit. Its output goes where ours does (the boot
    log), so a start that fails says why there."""
    if sys.platform != "win32":
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = 10000 if hard == resource.RLIM_INFINITY else min(10000, hard)
        if soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    command = [*RAY_CLI, "start", *args]
    print(f'### Starting Ray with {" ".join(f"{k}={v}" for k, v in env.items())} {shlex.join(command)}')
    return subprocess.Popen(command, env={**os.environ, **RAY_ENV, **env}, creationflags=CHILD_FLAGS)


# Generous: under x64 emulation on an ARM Windows VM a start takes well over a
# minute; a start that fails is caught by its exit code, not by this.
_START_TIMEOUT_S = 180


def start_ray_head_node(node_ip_address: str | None = None, timeout=_START_TIMEOUT_S):
    # TODO: Check for an existing Ray head node.
    # TODO: Put these resources in the UI.
    # For debug-style one-task-at-a-time serialization, add {"fake_resource": 1}
    # here and tag the target functions with resources={"fake_resource": 1}.
    resources = _resources({"PixInsight": 1, "diskcache": 1, "fastest_storage": 1, "gpu_slots": 3})
    bind = ["--node-ip-address", node_ip_address] if node_ip_address else []
    _fresh_temp_dir()
    process = _start_ray(
        ["--head", *bind, "--memory", str(_ray_memory_bytes()), "--resources", resources],
        {"RAY_num_workers_soft_limit": str(_workers_soft_limit())})

    try:
        _wait_for_ray(process, timeout)
    except (TimeoutError, RuntimeError):
        process.kill()
        raise

    return process


def get_ip() -> str:
    """Retrieve the local machine's LAN IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def start_ray_subordinate_node(head_node_address, node_ip_address: str | None = None,
                               timeout=_START_TIMEOUT_S):
    # JOIN FENCE: refuse to join when this process context cannot read the
    # shared volumes. A worker that joins blind LOOKS healthy to the
    # scheduler but fails every file-touching task it is handed — three
    # separate blind-rejoin incidents burned ~40 nodes on 2026-08-20/21
    # (sshd without Full Disk Access, an iTerm-launched follower without
    # the TCC grant, and a half-unmounted ghost /Volumes/AstroRaw-1).
    # macOS TCC is PER LAUNCHING APP, so only a runtime probe in THIS
    # process can know. Fail loud, join never.
    import os
    for volume in ("/Volumes/AstroRaw", "/Volumes/AstroProcessed"):
        try:
            entries = os.listdir(volume)
        except OSError as e:
            raise RuntimeError(
                f"REFUSING to join the cluster: cannot read {volume} from "
                f"this process ({e}). Launch from a context with disk "
                f"access (Terminal.app, or grant the launcher Full Disk "
                f"Access) — a blind worker poisons every file-touching "
                f"task it is handed."
            ) from e
        # An UNMOUNTED volume is an empty directory, and listdir on it
        # succeeds — which is exactly how a blind worker slipped past the
        # first version of this fence on 2026-08-22 and failed ~20
        # placements. A shared data volume is never legitimately empty.
        if not entries:
            raise RuntimeError(
                f"REFUSING to join the cluster: {volume} is EMPTY, which "
                f"means it is not mounted (an unmounted mount point is a "
                f"bare directory that reads as empty). Mount it and "
                f"relaunch — a blind worker poisons every file-touching "
                f"task it is handed."
            )
    # Reap any local raylet BEFORE starting the one true node, so a join is
    # idempotent: exactly one raylet per machine, always. A worker process can
    # exit (crash, manual restart, version-change teardown) without stopping
    # Ray — the raylet the previous process started is then orphaned, and
    # `ray start` does not detect/replace it. Without this reap,
    # every code-follow (or crash-restart on a head bounce → WrongClusterID)
    # STACKS another raylet; observed 11 raylets on one 10-core worker, each
    # advertising 10 CPU + all of RAM, defeating CPU- and memory-gated scheduling
    # ~11×. `stop --force` is a fast, unconditional local reap (no-op when none
    # are running); only ever called on a worker joining, never on the head.
    subprocess.run([*RAY_CLI, "stop", "--force"], capture_output=True, text=True)
    _fresh_temp_dir()
    # TODO: Put these resources in the UI.
    resources = _resources({"PixInsight": 1, "gpu_slots": 2})
    bind = node_ip_address or get_ip()
    # Ray refuses multi-node clusters on macOS without the opt-in; set here, not
    # in a shell profile (fish universals are invisible to zsh/ssh launches).
    process = _start_ray(
        ["--address", f"{head_node_address}:6379", "--node-ip-address", bind,
         "--memory", str(_ray_memory_bytes()), "--resources", resources],
        {"RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER": "1",
         "RAY_num_workers_soft_limit": str(_workers_soft_limit())})

    try:
        _wait_for_ray(process, timeout)
    except (TimeoutError, RuntimeError):
        process.kill()
        raise

    return process


def stop_ray_node(timeout=15):
    subprocess.run([*RAY_CLI, "stop"], capture_output=True, text=True)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        print("### Waiting for Ray node to stop...")
        if not is_ray_connected():
            return
        time.sleep(1)

    print("### Ray did not stop in time, forcing...")
    subprocess.run([*RAY_CLI, "stop", "--force"], capture_output=True, text=True)
