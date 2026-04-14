import os
import subprocess
import time
import socket

from base_path import get_bin_dir


def is_ray_connected():
    """Check if the local Ray node is connected to a live cluster."""
    ray_path = get_bin_dir() / "ray"
    result = subprocess.run([ray_path, "status"], capture_output=True, text=True)
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


def start_ray_head_node(timeout=60):
    # TODO: Check for an existing Ray head node.
    ray_path = get_bin_dir() / "ray"
    # TODO: Put these resources in the UI.
    # For debug-style one-task-at-a-time serialization, add {"fake_resource": 1}
    # here and tag the target functions with resources={"fake_resource": 1}.
    resources = """'{"PixInsight": 1, "diskcache": 1, "fastest_storage": 1}'"""
    command = f"""ulimit -n 10000 && {ray_path} start --head --resources {resources}"""
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


def start_ray_subordinate_node(head_node_address, timeout=60):
    ray_path = get_bin_dir() / "ray"
    # TODO: Put these resources in the UI.
    resources = """'{"PixInsight": 1}'"""
    command = f"""ulimit -n 10000 && {ray_path} start --address {head_node_address}:6379 --node-ip-address {get_ip()} --resources {resources}"""
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
