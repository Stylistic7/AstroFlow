import os
import subprocess
import time
import socket

import psutil

from base_path import get_bin_dir


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
    """Memory to advertise to Ray as the node's static `memory` resource.

    Ray's default at startup is `available × 0.7`, taken once — so if
    PixInsight (or anything else heavy) was running when the head started,
    the cluster ends up with a permanently smaller cap than the machine
    can actually deliver. Anchoring on `total × 0.7` instead gives a
    stable per-machine number that doesn't depend on what was running
    at start time. The auto-memory system gates per-task scheduling, so
    a generous static cap doesn't risk OOM on its own.
    """
    return int(psutil.virtual_memory().total * 0.7)


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
    ray_path = get_bin_dir() / "ray"
    try:
        result = subprocess.run(
            [ray_path, "status"], capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return False
    return "Autoscaler status" in result.stdout


def _wait_for_ray(timeout):
    """Poll ray status until the node is up, or raise TimeoutError."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        print("### Waiting for Ray node to start...")
        if is_ray_connected():
            return
        time.sleep(1)
    raise TimeoutError(f"Ray node did not start within {timeout}s")


def _resources(base: dict) -> str:
    """The node's custom Ray resources as `ray start --resources` wants them.
    `Siril: 1` marks a node with Siril installed: integrations request a
    sliver of it, so it gates where they run, not how many."""
    import json
    from stacks.stacking.siril import SIRIL_APP
    if os.path.exists(os.environ.get("ASTROFLOW_SIRIL_BIN") or SIRIL_APP):
        base = {**base, "Siril": 1}
    return "'" + json.dumps(base) + "'"


def start_ray_head_node(node_ip_address: str | None = None, timeout=60):
    # TODO: Check for an existing Ray head node.
    ray_path = get_bin_dir() / "ray"
    # TODO: Put these resources in the UI.
    # For debug-style one-task-at-a-time serialization, add {"fake_resource": 1}
    # here and tag the target functions with resources={"fake_resource": 1}.
    resources = _resources({"PixInsight": 1, "diskcache": 1, "fastest_storage": 1, "gpu_slots": 3})
    bind = f" --node-ip-address {node_ip_address}" if node_ip_address else ""
    command = f"""ulimit -n 10000 && RAY_num_workers_soft_limit={_workers_soft_limit()} {ray_path} start --head{bind} --memory {_ray_memory_bytes()} --resources {resources}"""
    print(f'### Starting Ray head node with command "{command}"')
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line-buffered output
    )

    try:
        _wait_for_ray(timeout)
    except TimeoutError:
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


def start_ray_subordinate_node(head_node_address, node_ip_address: str | None = None, timeout=60):
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
    ray_path = get_bin_dir() / "ray"
    # Reap any local raylet BEFORE starting the one true node, so a join is
    # idempotent: exactly one raylet per machine, always. A worker process can
    # exit (crash, manual restart, version-change teardown) without stopping
    # Ray — the raylet the previous process started is then orphaned, and
    # `ray start` does not detect/replace it. Without this reap,
    # every code-follow (or crash-restart on a head bounce → WrongClusterID)
    # STACKS another raylet; observed 11 raylets on one 10-core worker, each
    # advertising 10 CPU + 0.7×RAM, defeating CPU- and memory-gated scheduling
    # ~11×. `stop --force` is a fast, unconditional local reap (no-op when none
    # are running); only ever called on a worker joining, never on the head.
    subprocess.run([ray_path, "stop", "--force"], capture_output=True, text=True)
    # TODO: Put these resources in the UI.
    resources = _resources({"PixInsight": 1, "gpu_slots": 2})
    bind = node_ip_address or get_ip()
    # Ray refuses multi-node clusters on macOS without the opt-in; set here, not
    # in a shell profile (fish universals are invisible to zsh/ssh launches).
    command = f"""ulimit -n 10000 && RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 RAY_num_workers_soft_limit={_workers_soft_limit()} {ray_path} start --address {head_node_address}:6379 --node-ip-address {bind} --memory {_ray_memory_bytes()} --resources {resources}"""
    print(f'### Starting Ray subordinate node with command "{command}"')
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line-buffered output
    )

    try:
        _wait_for_ray(timeout)
    except TimeoutError:
        process.kill()
        raise

    return process


def stop_ray_node(timeout=15):
    ray_path = get_bin_dir() / "ray"
    subprocess.run([ray_path, "stop"], capture_output=True, text=True)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        print("### Waiting for Ray node to stop...")
        if not is_ray_connected():
            return
        time.sleep(1)

    print("### Ray did not stop in time, forcing...")
    subprocess.run([ray_path, "stop", "--force"], capture_output=True, text=True)
