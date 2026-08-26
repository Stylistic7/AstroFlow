"""Worker-side head attachment: mount the head's shared disks over 10 GbE
and read the head's advertised code version.

Code does NOT move through here — code moves via git, and the worker FENCES
on version equality before joining (`state_machine._join_head`). This module
is mounts + one HTTP version read.

Everything here is worker-initiated. The head only *advertises* (via zeroconf)
its 10 GbE IP, its share names, and its `cluster_version`; nothing in this
module runs on the head, and the head never reaches into a worker.

Mounts go to `/Volumes/<share>` — the SAME absolute path the head uses — so the
content-addressed cache and on-disk file paths agree cluster-wide. `/Volumes`
is root-owned, so we can't `mkdir` + `mount_smbfs` there as a normal user;
instead we use macOS's `mount volume` (the automounter, the same path Finder
takes), which creates the mountpoint and mounts as the user. The share is
addressed by EXPLICIT IP so the directly-connected 10 GbE route is used rather
than WiFi.
"""
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

HTTP_PORT = 8000
_MOUNT_CONFIRM_S = 8


def _mountpoint(share: str) -> Path:
    return Path("/Volumes") / share


def _is_mounted(mp: Path) -> bool:
    return os.path.ismount(mp)


def _mount_is_alive(mp: Path, timeout_s: float = 5.0) -> bool:
    """True iff `mp` answers a real read within `timeout_s`.

    `os.path.ismount` only checks the mount TABLE. After a head reboot, a
    worker can hold a ZOMBIE SMB mountpoint from the dead session: still in
    the table, but every I/O errors or hangs — so the old
    ismount-then-continue path skipped remounting and the worker joined Ray
    with dead disks (WBPP then failed instantly on missing inputs,
    2026-08-10). Probe with a listdir in a worker thread because a dead SMB
    mount often HANGS rather than erroring — a plain listdir here could
    wedge the whole join loop."""
    import threading
    result: list[bool] = []

    def _probe():
        try:
            os.listdir(mp)
            result.append(True)
        except OSError:
            result.append(False)

    t = threading.Thread(target=_probe, daemon=True)
    t.start()
    t.join(timeout_s)
    return bool(result and result[0])


def _force_unmount(mp: Path) -> None:
    """Best-effort eviction of a dead mountpoint so a fresh mount can land
    at the CANONICAL path (macOS would otherwise mount at `<share>-1` and
    every task would read the stale/empty original)."""
    subprocess.run(["diskutil", "unmount", "force", str(mp)],
                   capture_output=True, text=True)


def mount_head_shares(head_ip: str, shares: list[str]) -> None:
    """Mount each `//head_ip/<share>` at `/Volumes/<share>` with NO explicit
    user in the URL — macOS then authenticates the current login session (same
    Apple ID across the LAN → passwordless). An explicit `user@` (or `guest@`)
    instead forces a per-user credential lookup that prompts / silently fails
    (the mount never appears). Verified: `mount volume "smb://<ip>/Astro"`
    mounts, `smb://nobody@<ip>/Astro` does not. Idempotent; raises if a mount
    can't be established."""
    for share in shares:
        mp = _mountpoint(share)
        if _is_mounted(mp):
            if _mount_is_alive(mp):
                continue
            # Zombie from a dead head session (post-reboot): evict and
            # fall through to a fresh mount at the canonical path.
            print(f"### {mp} is mounted but DEAD; force-unmounting to remount")
            _force_unmount(mp)
        url = f"smb://{head_ip}/{share}"
        result = subprocess.run(
            ["osascript", "-e", f'mount volume "{url}"'],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"mount volume {url} failed: {result.stderr.strip()}"
            )
        # `mount volume` blocks until mounted, but confirm defensively (SMB can
        # report success before the mount is visible).
        for _ in range(_MOUNT_CONFIRM_S):
            if _is_mounted(mp):
                break
            time.sleep(1)
        if not _is_mounted(mp):
            raise RuntimeError(f"{url} did not appear at {mp}")


def unmount_head_shares(shares: list[str]) -> None:
    """Best-effort force-unmount — used on quit and before remounting at a new
    head IP (a stale SMB mount otherwise blocks os.listdir for ~60 s)."""
    for share in shares:
        mp = _mountpoint(share)
        if _is_mounted(mp):
            subprocess.run(
                ["diskutil", "unmount", "force", str(mp)],
                capture_output=True,
            )


def head_current_version(head_ip: str) -> str:
    """The SHA the head is running RIGHT NOW, read from its AUTHORITATIVE HTTP
    endpoint. Zeroconf only tells us the head's IP — a killed prior head can leave
    a GHOST zeroconf record advertising a stale SHA, so we never follow that TXT;
    we ask the live head directly. Empty/'dev'/'wt:...' when the head isn't running
    committed code."""
    url = f"http://{head_ip}:{HTTP_PORT}/api/cluster-version"
    # Generous timeout: mid-rebuild the head's web server is CPU-starved, so even
    # this trivial read can spike from ~ms to tens of seconds.
    with urllib.request.urlopen(url, timeout=90) as r:
        return json.loads(r.read()).get("cluster_version", "")
